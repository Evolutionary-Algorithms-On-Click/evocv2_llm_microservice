"""FastAPI application for EVOC DEAP Agent - Modular version."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict
import logging

from app.models import SessionState
from app.config import settings
from app.routes import generate_router, modify_router, fix_router, sessions_router
from app.routes.generate import set_sessions_store as set_generate_sessions
from app.routes.modify import set_sessions_store as set_modify_sessions
from app.routes.fix import set_sessions_store as set_fix_sessions
from app.routes.sessions import set_sessions_store as set_sessions_sessions

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

    # Share session store across all route modules
    set_generate_sessions(sessions)
    set_modify_sessions(sessions)
    set_fix_sessions(sessions)
    set_sessions_sessions(sessions)

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
    No more repeated imports or inconsistent code across cells!

    ## Flexible Input
    Accepts flexible field structures - extra fields are captured as 'other_specifications' and sent to the LLM.
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    tags_metadata=[
        {
            "name": "generate",
            "description": "Generate new DEAP notebooks from specifications"
        },
        {
            "name": "modify",
            "description": "Modify existing notebooks with natural language"
        },
        {
            "name": "fix",
            "description": "Fix broken notebooks from error tracebacks"
        },
        {
            "name": "sessions",
            "description": "Manage notebook sessions"
        }
    ]
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
        "version": "2.0.0",
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
            "auto_fix": True,
            "other_specifications_support": True
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}


# Include routers
app.include_router(generate_router)
app.include_router(modify_router)
app.include_router(fix_router)
app.include_router(sessions_router)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main_new:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False
    )
