"""
Q&A phase handler with RAG (Retrieval-Augmented Generation) pipeline.

Pipeline: Question → Query Planning → Embed → Query VectorDB → Format Context → Generate Answer
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.models import ChatRequest, ChatResponse, Message, Source
from backend.services.openai_client import get_openai_client
from backend.services.vector_store import get_vector_store
from backend.prompts.qa_prompt import build_qa_prompt, format_retrieved_chunks
from backend.config import get_backend_settings

# Setup logging
logger = logging.getLogger(__name__)

# Query planning prompt
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


def clean_json_response(json_str: str) -> str:
    """
    Clean LLM JSON response by removing markdown code blocks and whitespace.

    LLMs sometimes wrap JSON in markdown blocks like:
    ```json
    {"key": "value"}
    ```

    Args:
        json_str: Raw JSON string from LLM

    Returns:
        Cleaned JSON string ready for parsing
    """
    # Remove markdown code blocks
    json_str = json_str.strip()

    # Remove ```json ... ``` wrapper
    if json_str.startswith("```json"):
        json_str = json_str[7:]  # Remove ```json
    elif json_str.startswith("```"):
        json_str = json_str[3:]  # Remove ```

    if json_str.endswith("```"):
        json_str = json_str[:-3]  # Remove trailing ```

    return json_str.strip()


async def plan_query(user_message: str, openai_client) -> Dict[str, Any]:
    """
    Use LLM to analyze the user's question and determine optimal query filters.

    This is the "Agentic RAG" approach - let the LLM decide which filters to apply
    instead of hardcoding keyword detection.

    Args:
        user_message: User's question
        openai_client: Azure OpenAI client

    Returns:
        Dictionary with query plan:
        {
            "chunk_type": str | None,
            "category": str | None,
            "ignore_tier": bool,
            "needs_comparison": bool
        }
    """
    try:
        # Call LLM for query planning (low temperature for consistency)
        response = await openai_client.chat(
            messages=[
                {"role": "system", "content": QUERY_PLANNING_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=150
        )

        json_str = response["content"].strip()

        # Clean JSON response (remove markdown code blocks)
        cleaned_json = clean_json_response(json_str)

        # Parse JSON
        query_plan = json.loads(cleaned_json)

        logger.info(f"Query plan: {query_plan}")
        return query_plan

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse query plan JSON: {e}. Response: {json_str}")
        # Fallback to empty plan (no special filtering)
        return {
            "chunk_type": None,
            "category": None,
            "ignore_tier": False,
            "needs_comparison": False
        }
    except Exception as e:
        logger.error(f"Query planning failed: {e}")
        # Fallback to empty plan
        return {
            "chunk_type": None,
            "category": None,
            "ignore_tier": False,
            "needs_comparison": False
        }


async def handle_qa_phase(request: ChatRequest) -> ChatResponse:
    """
    Handle the Q&A phase using RAG pipeline.

    RAG Pipeline Steps:
    1. Embed the user's question
    2. Query vector store with user's HMO and tier filters
    3. Format retrieved chunks as context
    4. Build prompt with user profile + context
    5. Generate answer using LLM
    6. Return response with sources

    Args:
        request: Chat request with user question, complete user_data, and history

    Returns:
        ChatResponse with answer, sources, and updated user_data
    """
    try:
        settings = get_backend_settings()

        # Get clients
        openai_client = get_openai_client()
        vector_store = get_vector_store()

        # Query Planning (LLM decides what filters to use)
        logger.info("Planning query...")
        query_plan = await plan_query(request.message, openai_client)

        # Embed the question
        logger.info("Embedding question...")
        question_embedding = await openai_client.embed(request.message)

        # Query vector store using the plan
        # Apply tier filter only if plan doesn't say to ignore it
        tier_filter = None if query_plan.get("ignore_tier") else request.user_data.tier

        logger.info(
            f"Querying vector store: hmo={request.user_data.hmo}, tier={tier_filter}, "
            f"type={query_plan.get('chunk_type')}, category={query_plan.get('category')}"
        )

        retrieved_chunks = vector_store.query(
            query_embedding=question_embedding,
            hmo=request.user_data.hmo,
            tier=tier_filter,
            chunk_type=query_plan.get("chunk_type"),
            category=query_plan.get("category"),
            n_results=settings.RAG_TOP_K
        )

        num_chunks = len(retrieved_chunks.get("documents", []))

        # Build retrieval strategy description
        strategy_parts = [f"hmo={request.user_data.hmo}"]
        if tier_filter:
            strategy_parts.append(f"tier={tier_filter}")
        if query_plan.get("chunk_type"):
            strategy_parts.append(f"type={query_plan['chunk_type']}")
        if query_plan.get("category"):
            strategy_parts.append(f"category={query_plan['category']}")

        retrieval_strategy = "planned (" + ", ".join(strategy_parts) + ")"

        logger.info(f"Retrieved {num_chunks} chunks using {retrieval_strategy}")

        # Fallback if no chunks found (try with fewer filters)
        if num_chunks == 0:
            logger.info("No results with planned filters, trying relaxed (hmo only)...")
            retrieved_chunks = vector_store.query(
                query_embedding=question_embedding,
                hmo=request.user_data.hmo,
                tier=None,
                chunk_type=None,
                category=None,
                n_results=settings.RAG_TOP_K
            )
            num_chunks = len(retrieved_chunks.get("documents", []))
            retrieval_strategy = "fallback (hmo only)"

        # Final fallback - global search
        if num_chunks == 0:
            logger.info("No results with fallback, trying global search...")
            retrieved_chunks = vector_store.query(
                query_embedding=question_embedding,
                hmo=None,
                tier=None,
                chunk_type=None,
                category=None,
                n_results=settings.RAG_TOP_K
            )
            num_chunks = len(retrieved_chunks.get("documents", []))
            retrieval_strategy = "global (no filters)"

        logger.info(f"Final retrieval: {num_chunks} chunks using {retrieval_strategy}")

        # Format retrieved chunks as context
        logger.info("Formatting context...")
        formatted_context = format_retrieved_chunks(retrieved_chunks, request.language)

        # Build system prompt with user profile + context
        logger.info("Building prompt...")
        system_prompt = build_qa_prompt(
            request.user_data,
            formatted_context,
            request.language
        )

        # Build messages for LLM
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Add conversation history
        for msg in request.conversation_history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        # Add current question
        messages.append({
            "role": "user",
            "content": request.message
        })

        # Generate answer using LLM
        logger.info("Generating answer...")
        response = await openai_client.chat(
            messages=messages,
            temperature=0.3,  # Lower temperature for factual answers
            max_tokens=800
        )

        answer = response["content"]
        tokens_used = response["tokens_used"]

        # Build sources list from retrieved chunks
        sources = build_sources_list(retrieved_chunks)

        logger.info(
            f"Q&A phase complete: {num_chunks} chunks retrieved, "
            f"{len(sources)} sources, tokens={tokens_used}"
        )

        return ChatResponse(
            response=answer,
            phase="qa",
            user_data=request.user_data,  # Return unchanged user data
            missing_fields=[],
            sources=sources,
            metadata={
                "tokens_used": tokens_used,
                "chunks_retrieved": num_chunks,
                "top_k": settings.RAG_TOP_K,
                "retrieval_strategy": retrieval_strategy
            }
        )

    except Exception as e:
        logger.error(f"Q&A phase error: {e}")
        raise


def build_sources_list(retrieved_chunks: Dict[str, Any]) -> List[Source]:
    """
    Build list of sources from retrieved chunks.

    Args:
        retrieved_chunks: Dictionary with documents, metadatas, and distances

    Returns:
        List of Source objects
    """
    sources = []

    if not retrieved_chunks or not retrieved_chunks.get("metadatas"):
        return sources

    metadatas = retrieved_chunks["metadatas"]
    distances = retrieved_chunks.get("distances", [])

    for i, metadata in enumerate(metadatas):
        # Calculate relevance score (distance to similarity)
        # ChromaDB uses cosine distance (0 = identical, 2 = opposite)
        # Convert to similarity score (0-1, where 1 = most similar)
        distance = distances[i] if i < len(distances) else 1.0
        relevance_score = max(0.0, 1.0 - (distance / 2.0))

        source = Source(
            type=metadata.get("type", "unknown"),
            category=metadata.get("category", "unknown"),
            service=metadata.get("service"),
            hmo=metadata.get("hmo", "unknown"),
            tier=metadata.get("tier", "unknown"),
            relevance_score=round(relevance_score, 3)
        )

        sources.append(source)

    # Sort by relevance score (highest first)
    sources.sort(key=lambda x: x.relevance_score, reverse=True)

    return sources
