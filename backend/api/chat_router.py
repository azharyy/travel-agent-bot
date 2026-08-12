# backend/api/chat_router.py
import logging
from fastapi import APIRouter, HTTPException, Depends, Request, status
from backend.core.schemas import ChatRequest, ChatResponse
from backend.core.config import Settings, get_settings

# Configure specialized logging for the routing ingestion layer
logger = logging.getLogger(__name__)

# Define the router with clear tagging
router = APIRouter(tags=["Chat Routing Pipeline"])


# FIX: Changed from "" to "/chat" to gracefully combine with main.py prefix="/api"
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request_body: ChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """
    Single entry point for all incoming travel agent conversational traffic.

    Delegates processing entirely to the central Orchestrator state machine 
    cached inside the FastAPI application state context during lifespan boot.
    No domain or business logic is managed here.
    """
    logger.info(f"Received inbound session payload for ID: {request_body.session_id}")

    try:
        # Resolve the warmed-up orchestrator instance from application state lifecycle
        orchestrator = request.app.state.orchestrator
        
        # Execute the unified 4-stage pipeline workflow
        result = await orchestrator.run(request_body)
        
        logger.info(f"Pipeline processing successfully completed for session: {request_body.session_id}")
        return result

    except AttributeError as ae:
        # Triggers cleanly if the orchestrator failed to cache in app.state on startup
        logger.critical(f"Orchestrator engine context missing from application state initialization lifecycle: {str(ae)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The core multi-agent orchestration service is currently uninitialized or warming up. Please try again shortly."
        )

    except Exception as e:
        # Fallback security layer isolating internal agent traces from leaking to the frontend client
        logger.error(f"Fatal pipeline extraction execution crash for session {request_body.session_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing pipeline workflows: {str(e)}"
        )