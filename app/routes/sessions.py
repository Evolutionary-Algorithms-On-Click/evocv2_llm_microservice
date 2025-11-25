"""Session management endpoints."""

from fastapi import APIRouter, HTTPException, status
import logging

from app.models import SessionState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/sessions", tags=["sessions"])

# Shared session store
sessions = {}


def set_sessions_store(store):
    """Set the global sessions store."""
    global sessions
    sessions = store


@router.get("/{session_id}", response_model=SessionState)
async def get_session(session_id: str):
    """
    Retrieve complete session state.

    Returns the current notebook and full operation history.
    """
    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )

    return sessions[session_id]


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )

    del sessions[session_id]
    logger.info(f"Deleted session {session_id}")

    return {"message": f"Session {session_id} deleted successfully"}


@router.get("")
async def list_sessions():
    """List all active sessions."""
    return {
        "sessions": [
            {
                "session_id": sid,
                "created_at": session.created_at,
                "updated_at": session.updated_at
            }
            for sid, session in sessions.items()
        ],
        "total": len(sessions)
    }
