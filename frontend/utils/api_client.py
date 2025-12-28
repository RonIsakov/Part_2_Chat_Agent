"""
REST API client for Medical Services Chatbot backend.

Handles all HTTP communication with the FastAPI backend.
"""

import requests
from typing import Dict, List, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatbotAPIClient:
    """
    Client for calling the Medical Services Chatbot REST API.

    Endpoints:
    - POST /api/v1/chat - Send message and get response
    - GET /api/v1/health - Check backend health
    """

    def __init__(self, backend_url: str = "http://localhost:8000"):
        """
        Initialize API client.

        Args:
            backend_url: Base URL of the FastAPI backend (default: http://localhost:8000)
        """
        self.backend_url = backend_url.rstrip("/")
        self.chat_endpoint = f"{self.backend_url}/api/v1/chat"
        self.health_endpoint = f"{self.backend_url}/api/v1/health"

        logger.info(f"API Client initialized with backend: {self.backend_url}")

    def send_message(
        self,
        message: str,
        user_data: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        language: str = "he"
    ) -> Optional[Dict[str, Any]]:
        """
        Send a chat message to the backend.

        Args:
            message: User's message
            user_data: User profile data (name, id, hmo, tier, etc.)
            conversation_history: List of previous messages
            language: Conversation language ("he" or "en")

        Returns:
            Response dictionary with:
            - response: Bot's reply
            - phase: Current phase ("collection" or "qa")
            - user_data: Updated user data
            - missing_fields: Fields still needed (collection phase)
            - sources: List of sources (Q&A phase)
            - metadata: Additional info (tokens, chunks, etc.)

        Raises:
            Exception: If request fails
        """
        try:
            # Build request payload
            payload = {
                "message": message,
                "user_data": user_data,
                "conversation_history": conversation_history,
                "language": language
            }

            logger.info(f"Sending message to {self.chat_endpoint}")

            # Send POST request
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                timeout=30  # 30 second timeout
            )

            # Check for HTTP errors
            response.raise_for_status()

            # Parse JSON response
            data = response.json()

            logger.info(f"Received response: phase={data.get('phase')}")

            return data

        except requests.exceptions.Timeout:
            logger.error("Request timed out (30s)")
            raise Exception("Request timed out. The server might be busy or offline.")

        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to backend at {self.backend_url}")
            raise Exception(
                f"Cannot connect to backend at {self.backend_url}. "
                "Make sure the backend server is running."
            )

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")

            # Try to get error details from response
            try:
                error_data = response.json()
                error_msg = error_data.get("detail", str(e))
            except:
                error_msg = str(e)

            raise Exception(f"Backend error: {error_msg}")

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Failed to send message: {str(e)}")

    def check_health(self) -> Dict[str, Any]:
        """
        Check backend health status.

        Returns:
            Health status dictionary with:
            - status: "healthy" or "degraded"
            - timestamp: Current time
            - components: Status of each component

        Raises:
            Exception: If health check fails
        """
        try:
            logger.info(f"Checking health at {self.health_endpoint}")

            response = requests.get(
                self.health_endpoint,
                timeout=5  # 5 second timeout for health check
            )

            response.raise_for_status()

            data = response.json()

            logger.info(f"Health check: {data.get('status')}")

            return data

        except requests.exceptions.Timeout:
            logger.error("Health check timed out")
            raise Exception("Health check timed out")

        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to backend")
            raise Exception("Backend is offline or unreachable")

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise Exception(f"Health check failed: {str(e)}")


# Singleton instance
_api_client = None


def get_api_client(backend_url: str = "http://localhost:8000") -> ChatbotAPIClient:
    """
    Get singleton API client instance.

    Args:
        backend_url: Backend URL (only used on first call)

    Returns:
        ChatbotAPIClient instance
    """
    global _api_client

    if _api_client is None:
        _api_client = ChatbotAPIClient(backend_url)

    return _api_client
