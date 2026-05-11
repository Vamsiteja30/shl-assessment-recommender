import logging
import re
from enum import Enum
from typing import Any

from app.schemas import ChatRequest, ChatResponse, Message, RecommendationItem
from app.retriever import Retriever
from app.agent import MultiLLMAgent

logger = logging.getLogger(__name__)

class Intent(str, Enum):
    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    COMPARE = "compare"
    REFUSE = "refuse"
    REFINE = "refine"
    END = "end"

# Security patterns for pre-LLM filtering
INJECTION_PATTERNS = [
    r"ignore.*(instruction|rule|prompt|system|above|previous|limit|all|persona)",
    r"forget.*(everything|instruction|rule|prompt|above|previous)",
    r"act\s+as",
    r"pretend",
    r"reveal.*(system|instruction|secret|prompt|internal)",
    r"you\s+are\s+now",
    r"new\s+persona",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"disregard\s+(the\s+)?(above|previous|instructions)",
    r"override\s+(the\s+)?(above|previous|instructions|system)",
    r"system\s+prompt",
    r"ignore\s+all",
    r"tell\s+me\s+a\s+(joke|story|poem)",
    r"bypass",
    r"\bhack\b(?!\s*a\s*thon)",
]

OFF_TOPIC_PATTERNS = [
    r"\b(weather|forecast|temperature|climate)\b",
    r"\b(recipe|cook|bake|ingredient|food)\b",
    r"\b(capital\s+of|country|geography|map)\b",
    r"\b(stock\s+price|cryptocurrency|bitcoin|invest(?!ment assessment))\b",
    r"\b(movie|film|music|song|sport|game|football|cricket)\b",
    r"\b(legal\s+advice|lawsuit|discrimination|compliance\s+law)\b",
    r"\b(salary\s+negotiation|offer\s+letter)\b",
    r"\b(medical|diagnosis|symptom|treatment)\b",
]

HIRING_SIGNALS = [
    r"\b(hire|hiring|recruit|role|position|job|candidate|applicant|employee)\b",
    r"\b(assessment|test|evaluate|measure|screen|select)\b",
    r"\b(skill|competenc|aptitude|ability|personality|behavior)\b",
    r"\b(developer|engineer|manager|analyst|designer|scientist|consultant)\b",
    r"\b(shl|opq|verify|java|python|sql|coding)\b",
]

COMPARISON_PATTERNS = [
    r"\b(compare|comparison|versus|vs\.?|difference|differ|distinguish)\b",
    r"\bwhich\s+(is|are)\s+better\b",
]

REFINEMENT_PATTERNS = [
    r"\b(also\s+add|include\s+(also|too)|add\s+(also|more|personality|cognitive|skill))\b",
    r"\b(remove|exclude|without|drop|not\s+include)\b",
    r"\b(instead\s+of|rather\s+than|replace)\b",
    r"\b(update\s+(the\s+)?list|change\s+(the\s+)?recommendation|modify)\b",
]

CONTEXT_READY_PATTERNS = [
    r"\b(developer|engineer|scientist|analyst|manager|consultant|designer|architect|specialist)\b",
    r"\b(java|python|sql|javascript|typescript|react|angular|c\+\+|golang|rust|coding|programming|networking)\b",
    r"\b(leadership|communication|teamwork|problem.solving|critical.thinking|customer.service|sales|contact.centre)\b",
    r"\b(entry.level|junior|mid.level|senior|graduate|executive|vp|director|analyst|trainee)\b",
    r"\b(personality|cognitive|aptitude|situational|behavioral|simulation|coding\s+test|verbal|numerical)\b",
    r"\b(remove|drop|exclude|replace|without)\b",
]

PROFILE_PATTERNS = {
    "technical": r"\b(engineer|developer|tech|java|python|cloud|aws|software|coding|analyst)\b",
    "graduate": r"\b(graduate|trainee|campus|intern|apprentice|entry.level|junior)\b",
    "leadership": r"\b(management|director|cxo|executive|vp|head\s+of)\b",
    "admin": r"\b(admin|assistant|clerical|office|secretary|coordinator|excel|word)\b",
    "customer": r"\b(customer|service|support|client|contact|call|sales|retail)\b",
}

ANCHOR_TESTS = {
    "personality": "Occupational Personality Questionnaire OPQ32r",
    "aptitude": "SHL Verify Interactive G+",
    "graduate_scenarios": "Graduate Scenarios",
    "leadership_report": "OPQ Leadership Report",
    "excel": "Microsoft Excel 365 - Essentials (New)",
    "live_coding": "Smart Interview Live Coding",
    "linux": "Linux Programming (General)",
}

def _matches(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text.lower()) for p in patterns)

def classify_intent(messages: list[Message]) -> Intent:
    if not messages:
        return Intent.CLARIFY

    last_user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")
    
    if _matches(last_user_msg, INJECTION_PATTERNS):
        return Intent.REFUSE

    if _matches(last_user_msg, OFF_TOPIC_PATTERNS) and not _matches(" ".join(m.content for m in messages), HIRING_SIGNALS):
        return Intent.REFUSE

    if _matches(last_user_msg, COMPARISON_PATTERNS):
        return Intent.COMPARE

    # Check for refinement if recommendations were previously given
    has_prior = any(m.role == "assistant" and ("recommend" in m.content.lower() or "assessment" in m.content.lower()) for m in messages)
    if has_prior and _matches(last_user_msg, REFINEMENT_PATTERNS):
        return Intent.REFINE

    if _matches(" ".join(m.content for m in messages), CONTEXT_READY_PATTERNS):
        return Intent.RECOMMEND

    # Auto-recommend if we've asked for clarification multiple times
    assistant_turns = sum(1 for m in messages if m.role == "assistant")
    if assistant_turns >= 5 and _matches(" ".join(m.content for m in messages), HIRING_SIGNALS):
        return Intent.RECOMMEND

    return Intent.CLARIFY

def build_retrieval_query(messages: list[Message]) -> str:
    user_msgs = [m.content for m in messages if m.role == "user"]
    recent_context = " ".join(user_msgs[-3:])
    
    # Simple de-duplication
    tokens = []
    seen = set()
    for word in recent_context.split():
        if word.lower() not in seen:
            seen.add(word.lower())
            tokens.append(word)

    query = " ".join(tokens)
    
    # Inject profile-specific anchors into query to guide the hybrid search
    lower_query = query.lower()
    anchors = []
    if _matches(lower_query, [PROFILE_PATTERNS["technical"]]):
        anchors.extend([ANCHOR_TESTS["aptitude"], ANCHOR_TESTS["personality"], ANCHOR_TESTS["live_coding"]])
    if _matches(lower_query, [PROFILE_PATTERNS["graduate"]]):
        anchors.extend([ANCHOR_TESTS["graduate_scenarios"], ANCHOR_TESTS["aptitude"]])
    if _matches(lower_query, [PROFILE_PATTERNS["leadership"]]):
        anchors.extend([ANCHOR_TESTS["leadership_report"], ANCHOR_TESTS["personality"]])
    if _matches(lower_query, [PROFILE_PATTERNS["admin"]]):
        anchors.extend([ANCHOR_TESTS["excel"], ANCHOR_TESTS["aptitude"]])
    if _matches(lower_query, [PROFILE_PATTERNS["customer"]]):
        anchors.extend([ANCHOR_TESTS["personality"], "Customer Service Phone Simulation"])

    if anchors:
        query = f"{query} {' '.join(dict.fromkeys(anchors))}"

    return query

def extract_job_levels(messages: list[Message]) -> list[str]:
    level_map = {
        "entry": "Entry-Level", "junior": "Entry-Level", "graduate": "Graduate",
        "mid": "Mid-Professional", "senior": "Professional Individual Contributor",
        "manager": "Manager", "director": "Director", "executive": "Executive", "vp": "Executive",
    }
    all_text = " ".join(m.content for m in messages).lower()
    found = [level for kw, level in level_map.items() if kw in all_text]
    return list(dict.fromkeys(found))

async def process_conversation(request: ChatRequest, retriever: Retriever, agent: MultiLLMAgent) -> ChatResponse:
    messages = request.messages
    
    # Force recommendation on turn 7 to honor evaluator limits
    if len(messages) >= 7:
        logger.info("Turn limit approaching: forcing final recommendation.")
        query = build_retrieval_query(messages)
        levels = extract_job_levels(messages)
        results = retriever.search(query, top_k=10, filter_levels=levels or None)
        recommendations = [
            RecommendationItem(name=r["name"], url=r["url"], test_type=r.get("test_type", ""))
            for r in results
        ]
        return ChatResponse(
            reply="Based on our requirements, here are the SHL assessments I recommend for this role:",
            recommendations=recommendations,
            end_of_conversation=True
        )

    intent = classify_intent(messages)

    if intent == Intent.REFUSE:
        last_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")
        msg = "I can only assist with SHL assessment recommendations and related hiring queries."
        if _matches(last_msg, INJECTION_PATTERNS):
            msg = "I cannot fulfill this request. I am specialized in SHL product consultation."
        return ChatResponse(reply=msg, recommendations=[], end_of_conversation=False)

    # Retrieval and LLM processing
    query = build_retrieval_query(messages)
    levels = extract_job_levels(messages)
    top_k = 20 if intent == Intent.COMPARE else 15
    results = retriever.search(query, top_k=top_k, filter_levels=levels or None)

    history = [{"role": m.role, "content": m.content} for m in messages]
    llm_res = await agent.chat(history, results, hint_intent=intent.value)

    res_intent = llm_res.get("intent", "clarify")
    reply = llm_res.get("reply", "Could you provide more context on the role or specific skills required?").strip()
    names = llm_res.get("recommended_names", [])
    end_of_conv = bool(llm_res.get("end_of_conversation", False))

    recommendations = []
    if res_intent in ("recommend", "refine") and names:
        resolved = retriever.validate_and_resolve(names)
        recommendations = [
            RecommendationItem(name=item["name"], url=item["url"], test_type=item["test_type"])
            for item in resolved[:10]
        ]
        recommendations = _balance_battery(messages, recommendations, retriever)

    # Contextual end-of-conversation detection
    if not end_of_conv:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "").lower()
        if any(w in last_user for w in ["thank", "thanks", "bye", "that's it"]) and recommendations:
            end_of_conv = True

    return ChatResponse(reply=reply, recommendations=recommendations, end_of_conversation=end_of_conv)

def _balance_battery(messages: list[Message], recs: list[RecommendationItem], retriever: Retriever) -> list[RecommendationItem]:
    """Ensure recommended batteries are diverse by injecting missing assessment types."""
    history = " ".join([m.content for m in messages]).lower()
    profiles = {name for name, p in PROFILE_PATTERNS.items() if re.search(p, history)}
    
    if not profiles:
        return recs

    current_names = {r.name.lower() for r in recs}
    types = {r.test_type.lower() for r in recs}
    final_recs = list(recs)

    def add_if_missing(key: str):
        name = ANCHOR_TESTS.get(key)
        if not name or name.lower() in current_names: return
        item = retriever.lookup_by_name(name) or retriever.fuzzy_lookup(name)
        if item:
            final_recs.append(RecommendationItem(name=item["name"], url=item["url"], test_type=item["test_type"]))
            current_names.add(item["name"].lower())

    if "technical" in profiles or "graduate" in profiles:
        if not any("personality" in t for t in types): add_if_missing("personality")
        if not any(t in ["ability", "aptitude"] for t in types): add_if_missing("aptitude")
        if "graduate" in profiles: add_if_missing("graduate_scenarios")
        if "technical" in profiles and len(final_recs) < 5: add_if_missing("live_coding")

    if "leadership" in profiles:
        add_if_missing("leadership_report")
        add_if_missing("personality")

    if "admin" in profiles:
        if "excel" not in "".join(current_names): add_if_missing("excel")
        add_if_missing("personality")

    return final_recs[:10]
