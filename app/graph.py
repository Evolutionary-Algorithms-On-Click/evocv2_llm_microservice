"""LangGraph workflow for notebook operations."""

from typing import TypedDict, Literal, Optional, List, Any, Dict
from langgraph.graph import StateGraph, END
from app.models import (
    GenerateRequest, ModifyRequest, FixRequest,
    NotebookStructure, SessionState
)
from app.agents.generator_new import NotebookGenerator
from app.agents.modifier import NotebookModifier
from app.agents.fixer import NotebookFixer, NotebookValidator
from app.memory import session_memory
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """Shared state for LangGraph workflow."""
    operation: Literal["generate", "modify", "fix"]
    session_id: str
    request: Optional[Any]
    notebook: Optional[NotebookStructure]
    changes_made: List[str]
    validation_passed: bool
    error: Optional[str]
    retry_count: int
    max_retries: int


class NotebookWorkflow:
    """LangGraph workflow for notebook operations."""

    def __init__(self):
        self.generator = NotebookGenerator()
        self.modifier = NotebookModifier()
        self.fixer = NotebookFixer()
        self.validator = NotebookValidator()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("modify", self._modify_node)
        workflow.add_node("fix", self._fix_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("save_memory", self._save_memory_node)

        # Add conditional edges
        workflow.add_conditional_edges(
            "generate",
            self._route_after_generate,
            {
                "validate": "validate",
                "error": END
            }
        )

        workflow.add_conditional_edges(
            "modify",
            self._route_after_modify,
            {
                "validate": "validate",
                "error": END
            }
        )

        workflow.add_conditional_edges(
            "fix",
            self._route_after_fix,
            {
                "validate": "validate",
                "retry": "fix",
                "error": END
            }
        )

        workflow.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {
                "save": "save_memory",
                "fix": "fix",
                "end": END
            }
        )

        workflow.add_edge("save_memory", END)

        # Set entry point based on operation
        workflow.set_conditional_entry_point(
            self._route_entry,
            {
                "generate": "generate",
                "modify": "modify",
                "fix": "fix"
            }
        )

        return workflow.compile()

    def _route_entry(self, state: WorkflowState) -> str:
        """Route to appropriate entry node."""
        return state["operation"]

    def _route_after_generate(self, state: WorkflowState) -> str:
        """Route after generation."""
        if state.get("error"):
            return "error"
        return "validate"

    def _route_after_modify(self, state: WorkflowState) -> str:
        """Route after modification."""
        if state.get("error"):
            return "error"
        return "validate"

    def _route_after_fix(self, state: WorkflowState) -> str:
        """Route after fix attempt."""
        if state.get("error"):
            return "error"

        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)

        if not state.get("validation_passed") and retry_count < max_retries:
            return "retry"

        return "validate"

    def _route_after_validate(self, state: WorkflowState) -> str:
        """Route after validation."""
        if state.get("validation_passed"):
            return "save"

        # If fix operation, already handled in fix routing
        if state["operation"] == "fix":
            return "end"

        # For generate/modify, try to auto-fix
        retry_count = state.get("retry_count", 0)
        if retry_count < 2:
            return "fix"

        return "end"

    def _generate_node(self, state: WorkflowState) -> WorkflowState:
        """Generate notebook node."""
        try:
            request: GenerateRequest = state["request"]
            logger.info(f"Generating notebook for session {request.session_id}")

            notebook = self.generator.generate(request)

            state["notebook"] = notebook
            state["changes_made"] = ["Generated complete 12-cell DEAP notebook"]
            state["error"] = None

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            state["error"] = str(e)

        return state

    def _modify_node(self, state: WorkflowState) -> WorkflowState:
        """Modify notebook node."""
        try:
            request: ModifyRequest = state["request"]
            logger.info(f"Modifying notebook: {request.instruction}")

            modified_notebook, changes = self.modifier.modify(request)

            state["notebook"] = modified_notebook
            state["changes_made"] = changes
            state["error"] = None

        except Exception as e:
            logger.error(f"Modification failed: {e}")
            state["error"] = str(e)

        return state

    def _fix_node(self, state: WorkflowState) -> WorkflowState:
        """Fix notebook node."""
        try:
            if state["operation"] == "fix":
                request: FixRequest = state["request"]
            else:
                # Auto-fix from validation failure
                request = FixRequest(
                    traceback=state.get("error", "Validation failed"),
                    current_notebook=state["notebook"],
                    context="Auto-fix after validation failure"
                )

            logger.info("Attempting to fix notebook")

            fixed_notebook, fixes, success = self.fixer.fix(request, max_retries=1)

            state["notebook"] = fixed_notebook
            state["changes_made"].extend(fixes)
            state["validation_passed"] = success
            state["retry_count"] = state.get("retry_count", 0) + 1

            if not success:
                state["error"] = "Fix validation failed"

        except Exception as e:
            logger.error(f"Fix failed: {e}")
            state["error"] = str(e)

        return state

    def _validate_node(self, state: WorkflowState) -> WorkflowState:
        """Validate notebook node."""
        try:
            notebook = state["notebook"]

            is_valid, errors = self.validator.validate(notebook)

            state["validation_passed"] = is_valid

            if not is_valid:
                error_msg = "; ".join(errors)
                logger.warning(f"Validation failed: {error_msg}")
                state["error"] = error_msg
            else:
                logger.info("Validation passed")
                state["error"] = None

        except Exception as e:
            logger.error(f"Validation error: {e}")
            state["validation_passed"] = False
            state["error"] = str(e)

        return state

    def _save_memory_node(self, state: WorkflowState) -> WorkflowState:
        """Save interaction to memory."""
        try:
            session_id = state["session_id"]
            operation = state["operation"]

            details = {
                "changes": state.get("changes_made", []),
                "validation_passed": state.get("validation_passed", False)
            }

            result = "success" if state.get("validation_passed") else "partial"

            session_memory.add_interaction(
                session_id=session_id,
                operation=operation,
                details=details,
                result=result
            )

            logger.info(f"Saved interaction to memory for session {session_id}")

        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

        return state

    def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute the workflow."""
        try:
            final_state = self.graph.invoke(state)
            return final_state
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            state["error"] = str(e)
            state["validation_passed"] = False
            return state


# Global workflow instance
workflow = NotebookWorkflow()
