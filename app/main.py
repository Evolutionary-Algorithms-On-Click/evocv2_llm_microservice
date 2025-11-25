"""FastAPI application for EVOC DEAP Agent."""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict
from datetime import datetime
import logging

from app.models import (
    GenerateRequest, GenerateResponse,
    ModifyRequest, ModifyResponse,
    FixRequest, FixResponse,
    SessionState, ErrorResponse
)
from app.graph import workflow, WorkflowState
from app.memory import session_memory
from app.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# In-memory session store (stateless - sessions managed by caller)
sessions: Dict[str, SessionState] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting EVOC DEAP Agent service")
    logger.info(f"Model: {settings.model_name}")
    yield
    logger.info("Shutting down EVOC DEAP Agent service")


app = FastAPI(
    title="EVOC DEAP Agent",
    description="""
    Production-ready service for generating and maintaining DEAP evolutionary algorithms.

    ## Features
    * **Generate** - Create complete 12-cell DEAP notebooks from specifications
    * **Modify** - Update existing notebooks with natural language instructions
    * **Fix** - Automatically repair broken notebooks from error tracebacks
    * **Session Management** - Track and manage notebook sessions

    ## Single-Pass LLM Architecture
    All 12 cells are generated in a single LLM call for consistency and efficiency.

    ## Flexible Input
    Accepts flexible field structures - extra fields are captured as 'other_specifications' and sent to the LLM.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["info"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "EVOC DEAP Agent",
        "version": "1.0.0",
        "status": "running",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json"
        },
        "endpoints": {
            "generate": "POST /v1/generate",
            "modify": "POST /v1/sessions/{session_id}/modify",
            "fix": "POST /v1/sessions/{session_id}/fix",
            "get_session": "GET /v1/sessions/{session_id}",
            "list_sessions": "GET /v1/sessions",
            "delete_session": "DELETE /v1/sessions/{session_id}"
        },
        "features": {
            "single_pass_generation": True,
            "flexible_input": True,
            "auto_fix": True
        }
    }


@app.get("/health", tags=["info"])
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/v1/generate", response_model=GenerateResponse, tags=["generate"])
async def generate_notebook(request: GenerateRequest):
    """
    Generate a new 12-cell DEAP notebook from specification.

    Creates a complete, functional DEAP evolutionary algorithm notebook
    following the strict 12-cell structure using a single-pass LLM generation.

    **Features:**
    - Single LLM pass for all 12 cells (consistent code, no repeated imports)
    - Flexible input structure (extra fields captured as 'other_specifications')
    - Automatic validation and fixing
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


@app.post("/v1/sessions/{session_id}/modify", response_model=ModifyResponse, tags=["modify"])
async def modify_notebook(session_id: str, request: ModifyRequest):
    """
    Modify an existing notebook using natural language instructions.

    Supports intelligent dependency detection and cascading updates.
    """
    try:
        logger.info(f"Received modify request for session {session_id}")

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


@app.post("/v1/sessions/{session_id}/fix", response_model=FixResponse, tags=["fix"])
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


@app.get("/v1/sessions/{session_id}", response_model=SessionState, tags=["sessions"])
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


@app.delete("/v1/sessions/{session_id}", tags=["sessions"])
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


@app.get("/v1/sessions", tags=["sessions"])
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


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False
    )
