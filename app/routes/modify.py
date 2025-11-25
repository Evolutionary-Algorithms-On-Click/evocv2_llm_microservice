"""Modify endpoint for updating existing notebooks."""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime
import logging

from app.models import ModifyRequest, ModifyResponse
from app.graph import workflow, WorkflowState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/sessions", tags=["modify"])

# Shared session store
sessions = {}


def set_sessions_store(store):
    """Set the global sessions store."""
    global sessions
    sessions = store


@router.post("/{session_id}/modify", response_model=ModifyResponse)
async def modify_notebook(session_id: str, request: ModifyRequest):
    """
    Modify an existing notebook using natural language instructions.

    Supports intelligent dependency detection and cascading updates.
    Optionally specify cell_type to target specific cells for modification.
    """
    try:
        logger.info(f"Received modify request for session {session_id}")
        if request.cell_type:
            logger.info(f"Targeting cell_type: {request.cell_type}")

        # Execute workflow
        state: WorkflowState = {
            "operation": "modify",
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

        if final_state.get("error") and not final_state.get("notebook"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Modification failed: {final_state['error']}"
            )

        notebook = final_state["notebook"]
        changes = final_state.get("changes_made", [])

        # Update session if exists
        if session_id in sessions:
            sessions[session_id].notebook = notebook
            sessions[session_id].history.append({
                "operation": "modify",
                "instruction": request.instruction,
                "cell_type": request.cell_type,
                "timestamp": datetime.utcnow().isoformat(),
                "changes": changes
            })
            sessions[session_id].updated_at = datetime.utcnow().isoformat()

        logger.info(f"Successfully modified notebook for session {session_id}")

        return ModifyResponse(
            session_id=session_id,
            notebook=notebook,
            changes_made=changes,
            message="Notebook modified successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Modify endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
