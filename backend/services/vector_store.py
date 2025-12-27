"""
ChromaDB vector store client for RAG retrieval.

Provides query functionality with metadata filtering for HMO and tier.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import chromadb
from chromadb.config import Settings as ChromaSettings

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import get_backend_settings

# Setup logging
logger = logging.getLogger(__name__)


class VectorStoreClient:
    """
    ChromaDB vector store client wrapper.

    Provides:
    - Automatic initialization on first use
    - Query with metadata filtering (HMO, tier, category, type)
    - Health check functionality
    """

    def __init__(self):
        """Initialize and connect to ChromaDB immediately."""
        self.settings = get_backend_settings()

        try:
            # Connect to persistent ChromaDB
            self.client = chromadb.PersistentClient(
                path=self.settings.VECTOR_DB_PATH,
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            # Get existing collection
            self.collection = self.client.get_collection(
                name=self.settings.VECTOR_DB_COLLECTION_NAME
            )

            count = self.collection.count()
            logger.info(
                f"Vector store initialized: collection '{self.settings.VECTOR_DB_COLLECTION_NAME}' "
                f"with {count} chunks"
            )

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def query(
        self,
        query_embedding: List[float],
        hmo: Optional[str] = None,
        tier: Optional[str] = None,
        category: Optional[str] = None,
        chunk_type: Optional[str] = None,
        n_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query vector store with optional metadata filters.

        All filters are combined with AND logic using $and operator.

        Args:
            query_embedding: Embedding vector to search for
            hmo: Filter by HMO (maccabi, meuhedet, clalit, or None for all)
            tier: Filter by tier (gold, silver, bronze, or None for all)
            category: Filter by category (dental, optometry, etc.)
            chunk_type: Filter by type (context, benefit, contact)
            n_results: Number of results to return (default from settings)

        Returns:
            Dictionary with:
                - documents: List of matching text chunks
                - metadatas: List of metadata dicts
                - distances: List of similarity distances (lower is better)
                - ids: List of chunk IDs
        """
        if n_results is None:
            n_results = self.settings.RAG_TOP_K

        # Build metadata filter with explicit AND logic
        where_conditions = []

        if hmo:
            # Filter by exact HMO or "all" (for context/contact chunks)
            where_conditions.append({"hmo": {"$in": [hmo, "all"]}})

        if tier:
            # Filter by exact tier or "all"
            where_conditions.append({"tier": {"$in": [tier, "all"]}})

        if category:
            where_conditions.append({"category": category})

        if chunk_type:
            where_conditions.append({"type": chunk_type})

        # Combine all conditions with explicit AND
        if len(where_conditions) == 0:
            where_filter = None
        elif len(where_conditions) == 1:
            where_filter = where_conditions[0]
        else:
            # Use $and operator for explicit AND logic
            where_filter = {"$and": where_conditions}

        try:
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            # Flatten results (query_embeddings returns nested lists)
            flattened_results = {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "ids": results["ids"][0] if results["ids"] else []
            }

            logger.debug(
                f"Query returned {len(flattened_results['documents'])} results "
                f"(filters: hmo={hmo}, tier={tier}, category={category}, type={chunk_type})"
            )

            return flattened_results

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def health_check(self) -> bool:
        """
        Check if vector store is accessible and has data.

        Returns:
            True if healthy, False otherwise
        """
        try:
            count = self.collection.count()

            if count == 0:
                logger.warning("Health check failed: collection is empty")
                return False

            logger.debug(f"Health check passed: {count} chunks available")
            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with collection stats
        """
        try:
            count = self.collection.count()

            # Get sample to analyze metadata distribution
            sample = self.collection.get(limit=count, include=["metadatas"])

            # Count by type
            type_counts = {}
            hmo_counts = {}
            tier_counts = {}
            category_counts = {}

            for metadata in sample["metadatas"]:
                # Count types
                chunk_type = metadata.get("type", "unknown")
                type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1

                # Count HMOs
                hmo = metadata.get("hmo", "unknown")
                hmo_counts[hmo] = hmo_counts.get(hmo, 0) + 1

                # Count tiers
                tier = metadata.get("tier", "unknown")
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

                # Count categories
                category = metadata.get("category", "unknown")
                category_counts[category] = category_counts.get(category, 0) + 1

            return {
                "status": "initialized",
                "total_chunks": count,
                "by_type": type_counts,
                "by_hmo": hmo_counts,
                "by_tier": tier_counts,
                "by_category": category_counts
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"status": "error", "error": str(e)}


# Singleton instance
_vector_store: Optional[VectorStoreClient] = None


def get_vector_store() -> VectorStoreClient:
    """
    Get or create the singleton vector store client instance.

    Automatically initializes on first call.

    Returns:
        VectorStoreClient instance

    Raises:
        Exception: If initialization fails
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreClient()
    return _vector_store
