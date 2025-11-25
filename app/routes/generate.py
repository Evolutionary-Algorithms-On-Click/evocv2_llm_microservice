"""Generate endpoint for creating new DEAP notebooks."""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime
import logging

from app.models import GenerateRequest, GenerateResponse, SessionState
from app.graph import workflow, WorkflowState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["generate"])

# Shared session store (imported from main)
sessions = {}


def set_sessions_store(store):
    """Set the global sessions store."""
    global sessions
    sessions = store


@router.post("/generate", response_model=GenerateResponse)
async def generate_notebook(request: GenerateRequest):
    """
    Generate a new 12-cell DEAP notebook from specification.

    Creates a complete, functional DEAP evolutionary algorithm notebook
    following the strict 12-cell structure.
    """
    try:
        logger.info(f"Received generate request for session {request.session_id}")

        # Execute workflow
        state: WorkflowState = {
            "operation": "generate",
            "session_id": request.session_id,
            "request": request,
            "notebook": None,
            "changes_made": [],
            "validation_passed": False,
            "error": None,
            "retry_count": 0,
            "max_retries": 3
        }

        final_state = workflow.execute(state)

        if final_state.get("error") and not final_state.get("notebook"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation failed: {final_state['error']}"
            )

        notebook = final_state["notebook"]

        # Store session
        session = SessionState(
            session_id=request.session_id,
            notebook=notebook,
            problem=request.problem,
            algorithm=request.algorithm,
            operators=request.operators,
            features=request.features,
            history=[{
                "operation": "generate",
                "timestamp": datetime.utcnow().isoformat(),
                "changes": final_state.get("changes_made", [])
            }],
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )
        sessions[request.session_id] = session

        logger.info(f"Successfully generated notebook for session {request.session_id}")

        return GenerateResponse(
            session_id=request.session_id,
            notebook=notebook,
            message="Notebook generated successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generate endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
