"""
Collection phase prompt for gathering user information.

Two-step pattern:
1. Extraction: Silent JSON extraction from user message
2. Generation: Friendly response based on validated data
"""

import json
from typing import Dict, Optional, Tuple, List
from backend.models import UserData


# System prompt for collection phase
COLLECTION_SYSTEM_PROMPT_HE = """אתה עוזר ידידותי ומקצועי שתפקידו לאסוף מידע מהמשתמש על מנת לספק מידע מדויק על שירותי בריאות.

## המידע שצריך לאסוף:
1. **שם מלא** (פרטי ומשפחה)
2. **מספר תעודת זהות** (בדיוק 9 ספרות)
3. **מין** (זכר/נקבה/אחר)
4. **גיל** (בין 0 ל-120)
5. **קופת חולים** (מכבי/מאוחדת/כללית)
6. **מספר כרטיס קופת חולים** (בדיוק 9 ספרות)
7. **מסלול ביטוח** (זהב/כסף/ארד)

## כללי התנהגות:
- **שאל שאלה אחת בכל פעם** - אל תציף את המשתמש במספר שאלות בבת אחת
- **היה ידידותי וקצר** - השתמש בשפה פשוטה וברורה
- **אמת נתונים** - אם המשתמש נתן מידע לא תקין, הסבר מה הבעיה בצורה עדינה ובקש שוב
- **אל תמשיך הלאה** אם יש שדה לא תקין - תקן אותו תחילה
- **סכם בסוף** - לאחר איסוף כל המידע, הצג סיכום ובקש אישור

## כללי אימות:
- **מספר תעודת זהות**: בדיוק 9 ספרות, ללא אותיות או תווים מיוחדים
- **מספר כרטיס קופת חולים**: בדיוק 9 ספרות, ללא אותיות או תווים מיוחדים
- **גיל**: מספר בין 0 ל-120
- **קופת חולים**: רק מכבי, מאוחדת או כללית
- **מסלול**: רק זהב, כסף או ארד

## דוגמאות לטיפול בשגיאות:
- אם מספר ת"ז הוא 8 ספרות: "מספר תעודת זהות חייב להכיל בדיוק 9 ספרות. מה מספר תעודת הזהות שלך?"
- אם מספר ת"ז מכיל אותיות: "מספר תעודת זהות חייב להכיל רק ספרות. מה מספר תעודת הזהות שלך?"
- אם קופת חולים לא תקינה: "אני יכול לעזור עם קופות החולים הבאות: מכבי, מאוחדת או כללית. באיזו קופת חולים אתה מבוטח?"

## סיום איסוף המידע:
לאחר שכל 7 השדות מלאים ותקינים:
1. הצג סיכום של כל המידע שנאסף
2. שאל: "האם כל הפרטים נכונים?"
3. אם המשתמש מאשר, כתוב **בדיוק**: "COLLECTION_COMPLETE"
4. אם המשתמש רוצה לתקן משהו, חזור לשדה הרלוונטי

זכור: אתה רק אוסף מידע בשלב זה. לא צריך לענות על שאלות על שירותים רפואיים עד שהמידע מלא."""

COLLECTION_SYSTEM_PROMPT_EN = """You are a friendly and professional assistant whose role is to collect user information to provide accurate healthcare service information.

## Information to Collect:
1. **Full name** (first and last)
2. **ID number** (exactly 9 digits)
3. **Gender** (male/female/other)
4. **Age** (between 0 and 120)
5. **HMO** (Maccabi/Meuhedet/Clalit)
6. **HMO card number** (exactly 9 digits)
7. **Insurance tier** (Gold/Silver/Bronze)

## Behavior Rules:
- **Ask one question at a time** - don't overwhelm the user with multiple questions
- **Be friendly and concise** - use simple, clear language
- **Validate data** - if the user provides invalid information, gently explain the issue and ask again
- **Don't move forward** if a field is invalid - fix it first
- **Summarize at the end** - after collecting all information, show a summary and ask for confirmation

## Validation Rules:
- **ID number**: exactly 9 digits, no letters or special characters
- **HMO card number**: exactly 9 digits, no letters or special characters
- **Age**: number between 0 and 120
- **HMO**: only Maccabi, Meuhedet, or Clalit
- **Tier**: only Gold, Silver, or Bronze

## Examples of Error Handling:
- If ID is 8 digits: "ID number must contain exactly 9 digits. What is your ID number?"
- If ID contains letters: "ID number must contain only digits. What is your ID number?"
- If HMO is invalid: "I can help with the following HMOs: Maccabi, Meuhedet, or Clalit. Which HMO are you insured with?"

## Completing Collection:
After all 7 fields are complete and valid:
1. Show a summary of all collected information
2. Ask: "Is all the information correct?"
3. If the user confirms, write **exactly**: "COLLECTION_COMPLETE"
4. If the user wants to correct something, return to the relevant field

Remember: You are only collecting information at this stage. Don't answer questions about medical services until the information is complete."""


def build_collection_prompt(user_data: UserData, language: str = "he") -> str:
    """
    Build the collection system prompt with current user data status.

    Args:
        user_data: Current user data (may be partial)
        language: Conversation language ("he" or "en")

    Returns:
        System prompt string with data status
    """
    # Base prompt based on language
    if language == "he":
        base_prompt = COLLECTION_SYSTEM_PROMPT_HE
    else:
        base_prompt = COLLECTION_SYSTEM_PROMPT_EN

    # Add current status
    missing_fields = user_data.get_missing_fields()

    if language == "he":
        status = "\n\n## סטטוס נוכחי:\n"
        status += f"שדות שנאספו: {7 - len(missing_fields)}/7\n"

        if missing_fields:
            status += f"שדות חסרים: {', '.join(missing_fields)}\n"
        else:
            status += "כל השדות נאספו! הצג סיכום ובקש אישור.\n"

        # Show collected data
        if user_data.name:
            status += f"- שם: {user_data.name}\n"
        if user_data.id:
            status += f"- ת.ז: {user_data.id}\n"
        if user_data.gender:
            status += f"- מין: {user_data.gender}\n"
        if user_data.age is not None:
            status += f"- גיל: {user_data.age}\n"
        if user_data.hmo:
            status += f"- קופת חולים: {user_data.hmo}\n"
        if user_data.hmo_card:
            status += f"- כרטיס קופת חולים: {user_data.hmo_card}\n"
        if user_data.tier:
            status += f"- מסלול: {user_data.tier}\n"

    else:
        status = "\n\n## Current Status:\n"
        status += f"Fields collected: {7 - len(missing_fields)}/7\n"

        if missing_fields:
            status += f"Missing fields: {', '.join(missing_fields)}\n"
        else:
            status += "All fields collected! Show summary and ask for confirmation.\n"

        # Show collected data
        if user_data.name:
            status += f"- Name: {user_data.name}\n"
        if user_data.id:
            status += f"- ID: {user_data.id}\n"
        if user_data.gender:
            status += f"- Gender: {user_data.gender}\n"
        if user_data.age is not None:
            status += f"- Age: {user_data.age}\n"
        if user_data.hmo:
            status += f"- HMO: {user_data.hmo}\n"
        if user_data.hmo_card:
            status += f"- HMO card: {user_data.hmo_card}\n"
        if user_data.tier:
            status += f"- Tier: {user_data.tier}\n"

    return base_prompt + status


# STEP 1: Extraction Prompt (JSON output only)
EXTRACTION_PROMPT = """Extract and normalize user information from the conversation and output ONLY valid JSON.

**IMPORTANT**: Analyze the last 2-3 turns of conversation for context. If the bot asked "How old are you?" and user said "30", understand that 30 refers to age.

Output format (include only fields mentioned):
{
  "name": "string or null",
  "id": "string or null",
  "gender": "string or null",
  "age": number or null,
  "hmo": "string or null",
  "hmo_card": "string or null",
  "tier": "string or null"
}

Normalization Rules:
- **HMO**: Normalize to lowercase English: "Maccabi"/"מכבי" → "maccabi", "Meuhedet"/"מאוחדת" → "meuhedet", "Clalit"/"כללית" → "clalit"
- **Tier**: Normalize to lowercase English: "Gold"/"זהב" → "gold", "Silver"/"כסף" → "silver", "Bronze"/"ארד" → "bronze"
- **Gender**: Normalize to lowercase English: "Male"/"זכר" → "male", "Female"/"נקבה" → "female", "Other"/"אחר" → "other"
- **ID/HMO card**: Extract only digits, remove spaces/dashes
- **Age**: Extract number only
- **Name**: Keep as provided (capitalized properly)

Only include fields that the user explicitly mentioned in this turn or recent context.
Return null for fields not mentioned.
Output ONLY the JSON object, no explanation.

Examples:
User: "My name is Ron Isakov"
Output: {"name": "Ron Isakov", "id": null, "gender": null, "age": null, "hmo": null, "hmo_card": null, "tier": null}

Bot: "How old are you?"
User: "30"
Output: {"name": null, "id": null, "gender": null, "age": 30, "hmo": null, "hmo_card": null, "tier": null}

User: "I'm with מכבי זהב"
Output: {"name": null, "id": null, "gender": null, "age": null, "hmo": "maccabi", "hmo_card": null, "tier": "gold"}

User: "My ID is 123-456-789"
Output: {"name": null, "id": "123456789", "gender": null, "age": null, "hmo": null, "hmo_card": null, "tier": null}"""


def build_generation_prompt(
    user_data: UserData,
    validation_errors: Dict[str, str],
    language: str = "he"
) -> str:
    """
    Build the generation prompt for Step 2 (friendly response).

    Args:
        user_data: Current validated user data
        validation_errors: Dict of field_name -> error_message (can have multiple errors)
        language: Conversation language

    Returns:
        System prompt for generation
    """
    if language == "he":
        prompt = """אתה עוזר ידידותי שאוסף מידע מהמשתמש.

## המצב הנוכחי:
"""
        # Show current data
        if user_data.name:
            prompt += f"✓ שם: {user_data.name}\n"
        if user_data.id:
            prompt += f"✓ ת.ז: {user_data.id}\n"
        if user_data.gender:
            prompt += f"✓ מין: {user_data.gender}\n"
        if user_data.age is not None:
            prompt += f"✓ גיל: {user_data.age}\n"
        if user_data.hmo:
            prompt += f"✓ קופת חולים: {user_data.hmo}\n"
        if user_data.hmo_card:
            prompt += f"✓ כרטיס קופת חולים: {user_data.hmo_card}\n"
        if user_data.tier:
            prompt += f"✓ מסלול: {user_data.tier}\n"

        # Show validation errors (can be multiple)
        if validation_errors:
            prompt += "\n## שגיאות אימות:\n"
            for field, error in validation_errors.items():
                prompt += f"- {field}: {error}\n"

        # Instructions
        missing = user_data.get_missing_fields()
        if validation_errors:
            prompt += "\n**הוראות**: הסבר בעדינות את כל הבעיות ובקש את השדות שוב.\n"
        elif missing:
            prompt += f"\n**הוראות**: שאל על השדה החסר הבא: {missing[0]}\n"
        else:
            prompt += "\n**הוראות**: הצג סיכום ושאל אישור. אם המשתמש מאשר, כתוב 'COLLECTION_COMPLETE'\n"

    else:
        prompt = """You are a friendly assistant collecting user information.

## Current Status:
"""
        # Show current data
        if user_data.name:
            prompt += f"✓ Name: {user_data.name}\n"
        if user_data.id:
            prompt += f"✓ ID: {user_data.id}\n"
        if user_data.gender:
            prompt += f"✓ Gender: {user_data.gender}\n"
        if user_data.age is not None:
            prompt += f"✓ Age: {user_data.age}\n"
        if user_data.hmo:
            prompt += f"✓ HMO: {user_data.hmo}\n"
        if user_data.hmo_card:
            prompt += f"✓ HMO card: {user_data.hmo_card}\n"
        if user_data.tier:
            prompt += f"✓ Tier: {user_data.tier}\n"

        # Show validation errors (can be multiple)
        if validation_errors:
            prompt += "\n## Validation Errors:\n"
            for field, error in validation_errors.items():
                prompt += f"- {field}: {error}\n"

        # Instructions
        missing = user_data.get_missing_fields()
        if validation_errors:
            prompt += "\n**Instructions**: Gently explain all issues and ask for the fields again.\n"
        elif missing:
            prompt += f"\n**Instructions**: Ask for the next missing field: {missing[0]}\n"
        else:
            prompt += "\n**Instructions**: Show summary and ask for confirmation. If user confirms, write 'COLLECTION_COMPLETE'\n"

    return prompt
