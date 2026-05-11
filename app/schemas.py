from typing import List, Literal
from pydantic import BaseModel, field_validator

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("Conversation history cannot be empty")
        return v

class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[RecommendationItem]
    end_of_conversation: bool

class HealthResponse(BaseModel):
    status: str = "ok"
