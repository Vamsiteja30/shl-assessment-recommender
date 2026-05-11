# pyrefly: ignore [missing-import]
import streamlit as st 
import requests

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🏢",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── Sidebar / Config ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://www.shl.com/wp-content/uploads/SHL-Logo-Primary-RGB.svg", width=150)
    st.markdown("### Configuration")
    
    # Allow switching between local and deployed endpoints
    api_mode = st.radio("API Backend", ["Localhost", "Deployed (Render)"])
    
    if api_mode == "Localhost":
        api_url = st.text_input("API Base URL", value="http://localhost:8000")
    else:
        api_url = st.text_input("API Base URL", value="https://shl-assessment-recommender.onrender.com")

    st.markdown("---")
    if st.button("Reset Conversation", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.recommendations = []
        st.rerun()

# ── Session State Initialization ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

# ── Helper Functions ─────────────────────────────────────────────────────────
def get_type_badge(test_type: str) -> str:
    """Return a clean badge for the assessment type based on the SHL catalog."""
    test_type = test_type or "Knowledge & Skills"
    
    if "Personality" in test_type:
        return " Personality"
    elif "Ability" in test_type or "Aptitude" in test_type:
        return " Ability & Aptitude"
    elif "Situational" in test_type or "Biodata" in test_type:
        return " Situational"
    elif "Simulation" in test_type:
        return " Simulation"
    elif "Knowledge" in test_type or "Skill" in test_type:
        return " Knowledge & Skills"
    elif "Competencies" in test_type:
        return " Competencies"
    elif "Development" in test_type:
        return " Development"
    elif "Exercises" in test_type:
        return " Exercises"
    
    return f" {test_type}"

def render_recommendations(recs: list):
    """Render the recommendations nicely."""
    if not recs:
        return

    st.markdown("###  Recommended Assessments")
    for rec in recs:
        with st.container(border=True):
            name = rec.get("name", "Unknown Assessment")
            url = rec.get("url", "#")
            test_type = rec.get("test_type", "")
            
            badge = get_type_badge(test_type)
            
            st.markdown(f"**[{name}]({url})**")
            st.caption(f"**Type:** {badge}")

# ── Main Chat UI ─────────────────────────────────────────────────────────────
st.title("SHL Assessment Recommender")
st.markdown("Chat with the AI consultant to find the perfect SHL assessments for your hiring needs.")

# 1. Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. Display existing recommendations at the bottom of the history
if st.session_state.recommendations:
    render_recommendations(st.session_state.recommendations)

# 3. Chat Input
if prompt := st.chat_input("Describe the role you are hiring for..."):
    
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("Analyzing requirements..."):
            try:
                # Prepare payload ensuring we match the exact backend schema
                payload = {
                    "messages": st.session_state.messages
                }
                
                # Make the request to the FastAPI /chat endpoint
                response = requests.post(
                    f"{api_url.rstrip('/')}/chat",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    reply = data.get("reply", "I encountered an issue processing that.")
                    recommendations = data.get("recommendations", [])
                    
                    # Display the reply
                    st.markdown(reply)
                    
                    # Display recommendations if any
                    if recommendations:
                        render_recommendations(recommendations)
                    
                    # Save to session state
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.session_state.recommendations = recommendations
                    
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.Timeout:
                st.error("The request timed out. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to the API at {api_url}. Is the backend running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
