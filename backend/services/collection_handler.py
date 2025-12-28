"""
Collection phase handler with two-step pattern.

Step 1: Extract structured data (JSON) from user message
Step 2: Generate friendly response based on validated data
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.models import ChatRequest, ChatResponse, Message, UserData
from backend.services.openai_client import get_openai_client
from backend.prompts.collection_prompt import (
    EXTRACTION_PROMPT,
    build_generation_prompt
)

# Setup logging
logger = logging.getLogger(__name__)


async def extract_user_data(
    user_message: str,
    conversation_history: list,
    openai_client
) -> Dict[str, Any]:
    """
    STEP 1: Extract structured data from user message using LLM.

    Args:
        user_message: Current user message
        conversation_history: Last 2-3 turns for context
        openai_client: Azure OpenAI client

    Returns:
        Dictionary with extracted fields (normalized by LLM)
    """
    # Build messages with last 2-3 turns for context
    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT}
    ]

    # Add last 2-3 turns for context (helps with "30" → age mapping)
    recent_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
    for msg in recent_history:
        messages.append({
            "role": msg.role,
            "content": msg.content
        })

    # Add current message
    messages.append({
        "role": "user",
        "content": user_message
    })

    try:
        # Call LLM for extraction (low temperature for consistency)
        response = await openai_client.chat(
            messages=messages,
            temperature=0.1,
            max_tokens=200
        )

        json_str = response["content"].strip()

        # Parse JSON
        extracted_data = json.loads(json_str)

        logger.debug(f"Extracted data: {extracted_data}")
        return extracted_data

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse extraction JSON: {e}. Response: {json_str}")
        return {}
    except Exception as e:
        logger.error(f"Extraction step failed: {e}")
        return {}


def validate_and_merge(
    current_user_data: UserData,
    extracted_data: Dict[str, Any]
) -> Tuple[UserData, Dict[str, str]]:
    """
    Validate extracted data and merge with current user data.

    Args:
        current_user_data: Current user data state
        extracted_data: Newly extracted fields from LLM

    Returns:
        Tuple of (updated_user_data, validation_errors)
    """
    # Create a dict from current data
    updated_data = current_user_data.model_dump()

    # Merge extracted data (only non-null values)
    for field, value in extracted_data.items():
        if value is not None:
            updated_data[field] = value

    # Create new UserData instance (Pydantic will normalize)
    try:
        merged_user_data = UserData(**updated_data)
    except Exception as e:
        logger.error(f"Failed to create UserData: {e}")
        # Return current data unchanged if merge fails
        return current_user_data, {"general": "Failed to process data"}

    # Validate each updated field
    validation_errors = {}

    for field in extracted_data.keys():
        if extracted_data[field] is not None:
            is_valid, error_msg = merged_user_data.validate_field(field)
            if not is_valid:
                validation_errors[field] = error_msg

    logger.debug(f"Validation errors: {validation_errors}")

    # If there are validation errors, don't update those fields
    if validation_errors:
        # Revert invalid fields to previous values
        final_data = merged_user_data.model_dump()
        for field in validation_errors.keys():
            final_data[field] = getattr(current_user_data, field)

        final_user_data = UserData(**final_data)
        return final_user_data, validation_errors

    return merged_user_data, {}


async def generate_friendly_response(
    user_data: UserData,
    validation_errors: Dict[str, str],
    conversation_history: list,
    user_message: str,
    language: str,
    openai_client
) -> Tuple[str, int]:
    """
    STEP 2: Generate friendly response based on validated data.

    Args:
        user_data: Current validated user data
        validation_errors: Validation errors to explain
        conversation_history: Conversation history
        user_message: Current user message
        language: Conversation language
        openai_client: Azure OpenAI client

    Returns:
        Tuple of (friendly_response, tokens_used)
    """
    # Build generation prompt with validated state
    system_prompt = build_generation_prompt(user_data, validation_errors, language)

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Add conversation history
    for msg in conversation_history:
        messages.append({
            "role": msg.role,
            "content": msg.content
        })

    # Add current message
    messages.append({
        "role": "user",
        "content": user_message
    })

    # Call LLM for generation
    response = await openai_client.chat(
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )

    return response["content"], response["tokens_used"]


async def handle_collection_phase(request: ChatRequest) -> ChatResponse:
    """
    Handle the user information collection phase using two-step pattern.

    Step 1: Extract structured data (JSON)
    Step 2: Generate friendly response based on validated data

    Args:
        request: Chat request with current message, user_data, and history

    Returns:
        ChatResponse with LLM response, updated user_data, and collection status
    """
    try:
        # Check if this is the first message (empty conversation history)
        if len(request.conversation_history) == 0:
            logger.info("First message - sending introduction")

            # Return introduction message based on language
            if request.language == "he":
                greeting = """👋 שלום! אני העוזר הרפואי הדיגיטלי שלך.

אני כאן כדי לעזור לך למצוא מידע מדויק ומותאם אישית על שירותי בריאות של קופת החולים שלך.

לפני שנתחיל, אני צריך לאסוף כמה פרטים בסיסיים כדי לספק לך את המידע הרלוונטי ביותר.

בואו נתחיל - מה שמך המלא?"""
            else:
                greeting = """👋 Hello! I'm your digital medical assistant.

I'm here to help you find accurate and personalized information about your HMO's health services.

Before we begin, I need to collect some basic information so I can provide you with the most relevant details.

Let's get started - what is your full name?"""

            return ChatResponse(
                response=greeting,
                phase="collection",
                user_data=request.user_data,
                missing_fields=request.user_data.get_missing_fields(),
                sources=[],
                metadata={
                    "tokens_used": 0,
                    "fields_collected": 0,
                    "is_complete": False,
                    "is_greeting": True
                }
            )

        # Get OpenAI client
        openai_client = get_openai_client()

        # STEP 1: Extract structured data from user message
        logger.info("Collection Step 1: Extracting data...")
        extracted_data = await extract_user_data(
            request.message,
            request.conversation_history,
            openai_client
        )

        # Validate and merge with current data
        logger.info("Collection Step 1: Validating and merging...")
        updated_user_data, validation_errors = validate_and_merge(
            request.user_data,
            extracted_data
        )

        # STEP 2: Generate friendly response
        logger.info("Collection Step 2: Generating response...")
        friendly_response, tokens_used = await generate_friendly_response(
            updated_user_data,
            validation_errors,
            request.conversation_history,
            request.message,
            request.language,
            openai_client
        )

        # Check if collection is complete
        is_complete = "COLLECTION_COMPLETE" in friendly_response

        # Remove the marker from response if present
        if is_complete:
            friendly_response = friendly_response.replace("COLLECTION_COMPLETE", "").strip()
            # Set confirmed flag when user explicitly confirms
            updated_user_data.confirmed = True
            logger.info("User confirmed data - setting confirmed=True")

            # If response is empty after removing marker, add transition message
            if not friendly_response:
                if request.language == "he":
                    friendly_response = "מעולה! הפרטים שלך נשמרו. כעת אתה יכול לשאול אותי כל שאלה על שירותי בריאות של קופת החולים שלך."
                else:
                    friendly_response = "Perfect! Your information has been saved. You can now ask me any questions about your HMO's health services."

        # Get missing fields
        missing_fields = updated_user_data.get_missing_fields()

        logger.info(
            f"Collection phase complete: {len(missing_fields)} fields missing, "
            f"errors={len(validation_errors)}, complete={is_complete}, confirmed={updated_user_data.confirmed}, tokens={tokens_used}"
        )

        return ChatResponse(
            response=friendly_response,
            phase="collection",
            user_data=updated_user_data,  # Return updated state
            missing_fields=missing_fields,
            sources=[],
            metadata={
                "tokens_used": tokens_used,
                "fields_collected": 7 - len(missing_fields),
                "is_complete": is_complete,
                "validation_errors": list(validation_errors.keys()) if validation_errors else []
            }
        )

    except Exception as e:
        logger.error(f"Collection phase error: {e}")
        raise
