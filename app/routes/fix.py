"""Fix endpoint for repairing broken notebooks."""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime
import logging

from app.models import FixRequest, FixResponse
from app.graph import workflow, WorkflowState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/sessions", tags=["fix"])

# Shared session store
sessions = {}


def set_sessions_store(store):
    """Set the global sessions store."""
    global sessions
    sessions = store


@router.post("/{session_id}/fix", response_model=FixResponse)
async def fix_notebook(session_id: str, request: FixRequest):
    """
    Fix a broken notebook based on error traceback.

    Uses intelligent error analysis and validation loops to ensure fixes work.
    """
    try:
        logger.info(f"Received fix request for session {session_id}")

        # Execute workflow with retry loop
        state: WorkflowState = {
            "operation": "fix",
            "session_id": session_id,
            "request": request,
            "notebook": None,
            "changes_made": [],
            "validation_passed": False,
            "error": None,
            "retry_count": 0,
            "max_retries": 3
        }

        final_state = workflow.execute(state)

        notebook = final_state.get("notebook") or request.current_notebook
        fixes = final_state.get("changes_made", [])
        validation_passed = final_state.get("validation_passed", False)

        # Update session if exists
        if session_id in sessions:
            sessions[session_id].notebook = notebook
            sessions[session_id].history.append({
                "operation": "fix",
                "timestamp": datetime.utcnow().isoformat(),
                "fixes": fixes,
                "success": validation_passed
            })
            sessions[session_id].updated_at = datetime.utcnow().isoformat()

        message = "Notebook fixed successfully" if validation_passed else "Fixes applied but validation incomplete"

        logger.info(f"Fix completed for session {session_id}: {message}")

        return FixResponse(
            session_id=session_id,
            notebook=notebook,
            fixes_applied=fixes,
            validation_passed=validation_passed,
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fix endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
