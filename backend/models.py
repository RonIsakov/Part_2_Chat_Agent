"""
Pydantic models for Medical Chatbot API request/response validation.

Defines all data structures for the stateless microservice architecture.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


class UserData(BaseModel):
    """
    User information collected during the collection phase.

    All fields are optional and NO validation errors are raised.
    The LLM handles validation conversationally during collection phase.
    """

    name: Optional[str] = Field(None, description="First and last name")
    id: Optional[str] = Field(None, description="9-digit ID number")
    gender: Optional[str] = Field(None, description="Gender (male/female/other)")
    age: Optional[int] = Field(None, description="Age between 0 and 120")
    hmo: Optional[str] = Field(None, description="HMO name: maccabi, meuhedet, or clalit")
    hmo_card: Optional[str] = Field(None, description="9-digit HMO card number")
    tier: Optional[str] = Field(None, description="Insurance tier: gold, silver, or bronze")
    confirmed: bool = Field(default=False, description="Whether user has confirmed their information")

    @field_validator("hmo")
    @classmethod
    def normalize_hmo(cls, v: Optional[str]) -> Optional[str]:
        """Normalize HMO to English (no validation errors)."""
        if v is None:
            return v
        # Normalize to English for internal consistency
        hmo_map = {
            "מכבי": "maccabi",
            "meuhedet": "meuhedet",
            "מאוחדת": "meuhedet",
            "כללית": "clalit",
            "clalit": "clalit",
            "maccabi": "maccabi"
        }
        return hmo_map.get(v.strip(), v.strip().lower())

    @field_validator("tier")
    @classmethod
    def normalize_tier(cls, v: Optional[str]) -> Optional[str]:
        """Normalize tier to English (no validation errors)."""
        if v is None:
            return v
        # Normalize to English for internal consistency
        tier_map = {
            "זהב": "gold",
            "gold": "gold",
            "כסף": "silver",
            "silver": "silver",
            "ארד": "bronze",
            "bronze": "bronze"
        }
        return tier_map.get(v.strip(), v.strip().lower())

    @field_validator("gender")
    @classmethod
    def normalize_gender(cls, v: Optional[str]) -> Optional[str]:
        """Normalize gender to lowercase (no validation errors)."""
        if v is None:
            return v
        return v.strip().lower()

    def is_complete(self) -> bool:
        """Check if all required fields are filled."""
        return all([
            self.name,
            self.id,
            self.gender,
            self.age is not None,  # Preserve age 0 for infants
            self.hmo,
            self.hmo_card,
            self.tier
        ])

    def get_missing_fields(self) -> List[str]:
        """Get list of missing required fields."""
        missing = []
        if not self.name:
            missing.append("name")
        if not self.id:
            missing.append("id")
        if not self.gender:
            missing.append("gender")
        if self.age is None:  # Correctly handle age 0
            missing.append("age")
        if not self.hmo:
            missing.append("hmo")
        if not self.hmo_card:
            missing.append("hmo_card")
        if not self.tier:
            missing.append("tier")
        return missing

    def validate_field(self, field_name: str) -> tuple[bool, Optional[str]]:
        """
        Validate a specific field and return (is_valid, error_message).

        This is used by the LLM during collection to provide conversational feedback.

        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        if field_name == "id":
            if self.id:
                if not self.id.isdigit():
                    return False, "ID must contain only digits"
                if len(self.id) != 9:
                    return False, "ID must be exactly 9 digits"

        elif field_name == "hmo_card":
            if self.hmo_card:
                if not self.hmo_card.isdigit():
                    return False, "HMO card must contain only digits"
                if len(self.hmo_card) != 9:
                    return False, "HMO card must be exactly 9 digits"

        elif field_name == "age":
            if self.age is not None:
                if self.age < 0 or self.age > 120:
                    return False, "Age must be between 0 and 120"

        elif field_name == "hmo":
            if self.hmo:
                valid_hmos = {"maccabi", "meuhedet", "clalit"}
                if self.hmo.lower() not in valid_hmos:
                    return False, "HMO must be one of: Maccabi, Meuhedet, Clalit"

        elif field_name == "tier":
            if self.tier:
                valid_tiers = {"gold", "silver", "bronze"}
                if self.tier.lower() not in valid_tiers:
                    return False, "Tier must be one of: Gold, Silver, Bronze"

        elif field_name == "gender":
            if self.gender:
                valid_genders = {"male", "female", "other", "זכר", "נקבה", "אחר"}
                if self.gender.lower() not in valid_genders:
                    return False, "Gender must be one of: male, female, other"

        return True, None


class Message(BaseModel):
    """Single message in conversation history."""

    role: Literal["user", "assistant", "system"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """
    Request model for the chat endpoint.

    Stateless design: all context passed in each request.
    """

    message: str = Field(..., description="Current user message")
    user_data: UserData = Field(default_factory=UserData, description="User information")
    conversation_history: List[Message] = Field(
        default_factory=list,
        description="Previous conversation messages"
    )
    language: Literal["he", "en"] = Field(default="he", description="Conversation language")

    @model_validator(mode="after")
    def truncate_history(self):
        """
        Keep only last N messages to prevent token overflow.

        Uses MAX_CONVERSATION_HISTORY from BackendSettings.
        """
        from config import get_backend_settings

        max_history = get_backend_settings().MAX_CONVERSATION_HISTORY

        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]

        return self


class Source(BaseModel):
    """Source citation from RAG retrieval."""

    type: str = Field(..., description="Chunk type: context, benefit, or contact")
    category: str = Field(..., description="Medical service category")
    service: Optional[str] = Field(None, description="Service name (for benefit chunks)")
    hmo: str = Field(..., description="HMO name")
    tier: str = Field(..., description="Insurance tier")
    relevance_score: float = Field(..., description="Similarity score (0-1)")


class ChatResponse(BaseModel):
    """
    Response model for the chat endpoint.

    Contains LLM response plus metadata about the conversation state.
    Stateless design: returns updated user_data for client to store.
    """

    response: str = Field(..., description="LLM response message")
    phase: Literal["collection", "qa"] = Field(..., description="Current conversation phase")
    user_data: UserData = Field(
        default_factory=UserData,
        description="Updated user data (stateless - client must store and send back)"
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="List of missing user data fields (collection phase only)"
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="Source citations from knowledge base (Q&A phase only)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tokens used, retrieval info, etc.)"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Service health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of individual components (vector_store, azure_openai, etc.)"
    )
