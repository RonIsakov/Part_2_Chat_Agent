"""
Q&A phase prompt for answering questions using RAG retrieval.

Uses retrieved knowledge base chunks to provide accurate answers
specific to the user's HMO and insurance tier.
"""

from typing import Dict, List, Any
import sys
from pathlib import Path

# Add parent directory to path so we can import from root backend folder
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import UserData


# Query planning prompt for Agentic RAG
QUERY_PLANNING_PROMPT = """Analyze the user's question and determine the optimal database query filters.

Available chunk types:
- "benefit": Specific service benefits (discounts, coverage limits)
- "contact": Contact information (phone numbers, websites)
- "context": General background information

Available categories:
- "dental", "optometry", "alternative", "communication", "pregnancy", "workshops"

Output ONLY valid JSON with these fields:
{
  "chunk_type": "benefit" | "contact" | "context" | null,
  "category": "dental" | "optometry" | "alternative" | "communication" | "pregnancy" | "workshops" | null,
  "ignore_tier": true | false,
  "needs_comparison": true | false
}

Rules:
- Set "chunk_type": "contact" if user asks about phone numbers, calling, contacting, reaching out
- Set "ignore_tier": true for contact queries (contact info is tier-agnostic)
- Set "needs_comparison": true if comparing tiers (gold vs silver) or HMOs
- Set category if question is about a specific service type
- Set null for fields that shouldn't be filtered

Examples:

User: "What's Maccabi's phone number?"
Output: {"chunk_type": "contact", "category": null, "ignore_tier": true, "needs_comparison": false}

User: "How can I contact the dental department?"
Output: {"chunk_type": "contact", "category": "dental", "ignore_tier": true, "needs_comparison": false}

User: "How much is acupuncture?"
Output: {"chunk_type": "benefit", "category": "alternative", "ignore_tier": false, "needs_comparison": false}

User: "What's the difference between gold and silver for dental?"
Output: {"chunk_type": "benefit", "category": "dental", "ignore_tier": true, "needs_comparison": true}

User: "Tell me about alternative medicine"
Output: {"chunk_type": "context", "category": "alternative", "ignore_tier": true, "needs_comparison": false}

Output ONLY the JSON object, no explanation."""


def build_qa_prompt(
    user_data: UserData,
    retrieved_context: str,
    language: str = "he"
) -> str:
    """
    Build the Q&A system prompt with user profile and retrieved context.

    Args:
        user_data: Complete user information
        retrieved_context: Retrieved chunks from vector database (formatted string)
        language: Conversation language

    Returns:
        System prompt for Q&A
    """
    # Map internal values to Hebrew display names
    hmo_display = {
        "maccabi": "מכבי",
        "meuhedet": "מאוחדת",
        "clalit": "כללית"
    }

    tier_display = {
        "gold": "זהב",
        "silver": "כסף",
        "bronze": "ארד"
    }

    if language == "he":
        prompt = f"""אתה עוזר מומחה לשירותי בריאות שעונה על שאלות על בסיס בסיס הידע שסופק.

## פרופיל המשתמש:
- שם: {user_data.name}
- גיל: {user_data.age}
- מין: {user_data.gender}
- קופת חולים: {hmo_display.get(user_data.hmo, user_data.hmo)}
- מסלול ביטוח: {tier_display.get(user_data.tier, user_data.tier)}

## כללי התנהגות:
1. **ענה רק על בסיס המידע שסופק למטה** - אל תמציא מידע או תשתמש בידע כללי
2. **התמקד במסלול של המשתמש** - המשתמש במסלול {tier_display.get(user_data.tier, user_data.tier)} של {hmo_display.get(user_data.hmo, user_data.hmo)}
3. **צטט מספרים מדויקים** - אחוזי הנחה, מחירים, מגבלות - הכל חייב להיות מדויק
4. **אם אין מידע** - אמור בבירור "אין לי מידע על כך" במקום לנחש
5. **השווה בין מסלולים** - אם המשתמש שואל, הראה הבדלים בין זהב/כסף/ארד
6. **השווה בין קופות** - אם המשתמש שואל, הראה הבדלים בין מכבי/מאוחדת/כללית

## בסיס הידע (מידע רלוונטי שנמשך):
{retrieved_context}

## הוראות תשובה:
- היה ברור וקצר
- התחל עם המידע הרלוונטי ביותר למסלול של המשתמש
- אם יש מספרים (אחוזים, מחירים), ציין אותם במדויק
- אם המשתמש שואל "כמה עולה X?" - תן תשובה ספציפית למסלול שלו
- אם המשתמש שואל "מה ההבדל בין X ל-Y?" - השווה באופן ישיר

זכור: אתה משרת משתמש ב**{hmo_display.get(user_data.hmo, user_data.hmo)} {tier_display.get(user_data.tier, user_data.tier)}** - זה המידע החשוב ביותר עבורו!"""

    else:  # English
        prompt = f"""You are a medical services expert assistant that answers questions based on the provided knowledge base.

## User Profile:
- Name: {user_data.name}
- Age: {user_data.age}
- Gender: {user_data.gender}
- HMO: {user_data.hmo.title()}
- Insurance Tier: {user_data.tier.title()}

## Behavior Rules:
1. **Answer only based on the information provided below** - don't invent information or use general knowledge
2. **Focus on the user's tier** - The user has {user_data.tier.title()} tier with {user_data.hmo.title()}
3. **Quote exact numbers** - discounts, prices, limits - everything must be accurate
4. **If there's no information** - clearly say "I don't have information about that" instead of guessing
5. **Compare between tiers** - if the user asks, show differences between Gold/Silver/Bronze
6. **Compare between HMOs** - if the user asks, show differences between Maccabi/Meuhedet/Clalit

## Knowledge Base (retrieved relevant information):
{retrieved_context}

## Response Instructions:
- Be clear and concise
- Start with the most relevant information for the user's tier
- If there are numbers (percentages, prices), state them exactly
- If the user asks "How much does X cost?" - give a specific answer for their tier
- If the user asks "What's the difference between X and Y?" - compare directly

Remember: You're serving a user with **{user_data.hmo.title()} {user_data.tier.title()}** - this is the most important information for them!"""

    return prompt


def format_retrieved_chunks(chunks_dict: Dict[str, Any], language: str = "he") -> str:
    """
    Format retrieved chunks into readable context for the LLM.

    Args:
        chunks_dict: Dictionary with 'documents' and 'metadatas' keys from VectorStoreClient
        language: Conversation language

    Returns:
        Formatted context string
    """
    if not chunks_dict or not chunks_dict.get("documents"):
        if language == "he":
            return "לא נמצא מידע רלוונטי בבסיס הידע."
        else:
            return "No relevant information found in the knowledge base."

    documents = chunks_dict["documents"]
    metadatas = chunks_dict["metadatas"]

    context_parts = []

    # Display name mappings
    hmo_display_he = {
        "maccabi": "מכבי",
        "meuhedet": "מאוחדת",
        "clalit": "כללית",
        "all": "כל הקופות"  # Translate "all" to avoid confusion
    }

    tier_display_he = {
        "gold": "זהב",
        "silver": "כסף",
        "bronze": "ארד",
        "all": "כל המסלולים"  # Translate "all" to avoid confusion
    }

    for i, (doc, metadata) in enumerate(zip(documents, metadatas), 1):
        chunk_type = metadata.get("type", "unknown")
        category = metadata.get("category", "unknown")
        hmo = metadata.get("hmo", "unknown")
        tier = metadata.get("tier", "unknown")

        if language == "he":
            # Format based on chunk type
            if chunk_type == "context":
                context_parts.append(f"[הקשר כללי - {category}]\n{doc}\n")
            elif chunk_type == "benefit":
                service = metadata.get("service", "unknown")
                hmo_text = hmo_display_he.get(hmo, hmo)
                tier_text = tier_display_he.get(tier, tier)
                context_parts.append(
                    f"[שירות: {service} | קופה: {hmo_text} | מסלול: {tier_text}]\n{doc}\n"
                )
            elif chunk_type == "contact":
                hmo_text = hmo_display_he.get(hmo, hmo)
                context_parts.append(f"[פרטי התקשרות - {category} | {hmo_text}]\n{doc}\n")
            else:
                context_parts.append(f"[מידע]\n{doc}\n")
        else:  # English
            if chunk_type == "context":
                context_parts.append(f"[General Context - {category}]\n{doc}\n")
            elif chunk_type == "benefit":
                service = metadata.get("service", "unknown")
                hmo_text = "All HMOs" if hmo == "all" else hmo.title()
                tier_text = "All Tiers" if tier == "all" else tier.title()
                context_parts.append(
                    f"[Service: {service} | HMO: {hmo_text} | Tier: {tier_text}]\n{doc}\n"
                )
            elif chunk_type == "contact":
                hmo_text = "All HMOs" if hmo == "all" else hmo.title()
                context_parts.append(f"[Contact Info - {category} | {hmo_text}]\n{doc}\n")
            else:
                context_parts.append(f"[Information]\n{doc}\n")

    return "\n---\n".join(context_parts)
