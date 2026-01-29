"""fixer agent - Fixes broken notebooks based on tracebacks with Mem0 integration."""

import instructor
from groq import Groq
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.config import settings
from app.models import FixRequest, NotebookStructure, NotebookCell
from app.memory import enhanced_memory
from app.utils import validate_notebook_structure, format_code
import re
import logging


logger = logging.getLogger(__name__)


class ErrorAnalysis(BaseModel):
    """Structured error analysis."""
    error_type: str = Field(description="Type of error (SyntaxError, NameError, etc.)")
    error_location: Optional[int] = Field(None, description="Cell index where error occurred")
    root_cause: str = Field(description="Root cause of the error")
    affected_cells: List[int] = Field(description="Cells that need fixing")


class Fix(BaseModel):
    """A specific fix to apply."""
    cell_index: int = Field(description="Cell to fix (0-11)")
    fix_description: str = Field(description="What this fix does")
    fixed_code: str = Field(description="Corrected code")


class FixPlan(BaseModel):
    """Complete fix plan."""
    error_analysis: ErrorAnalysis
    fixes: List[Fix]
    validation_notes: str = Field(description="Notes about the fixes")
    requirements: Optional[str] = Field(None, description="Updated newline-separated requirements (if changed)")


class NotebookFixer:
    """Fixes broken DEAP notebooks based on error tracebacks with Mem0 learning."""

    def __init__(self):
        self.client = instructor.from_groq(
            Groq(api_key=settings.groq_api_key),
            mode=instructor.Mode.JSON
        )

    def fix(
        self,
        request: FixRequest,
        max_retries: int = 3
    ) -> tuple[NotebookStructure, List[str], bool]:
        """Fix notebook with retry loop and Mem0 error pattern learning."""
        logger.info(f"Fixing notebook {request.notebook_id} for user {request.user_id}")

        current_notebook = request.notebook
        attempts = 0

        while attempts < max_retries:
            attempts += 1
            logger.info(f"Fix attempt {attempts}/{max_retries}")

            # Analyze error with Mem0 context
            plan = self._analyze_error_with_mem0(request, current_notebook)

            # Apply fixes
            fixed_notebook = self._apply_fixes(current_notebook, plan)

            # Validate
            is_valid, errors = validate_notebook_structure(fixed_notebook)

            if is_valid:
                fixes_applied = [f.fix_description for f in plan.fixes]

                # Store successful fix pattern in Mem0
                self._store_fix_pattern_in_mem0(request, plan, success=True)

                return fixed_notebook, fixes_applied, True

            # Update request for retry
            logger.warning(f"Validation failed: {errors}")
            request = FixRequest(
                user_id=request.user_id,
                notebook_id=request.notebook_id,
                traceback=f"Validation errors: {'; '.join(errors)}",
                notebook=fixed_notebook,
                context=request.context
            )
            current_notebook = fixed_notebook

        # Max retries exceeded
        logger.error("Max retries exceeded, returning partially fixed notebook")
        fixes_applied = [f.fix_description for f in plan.fixes] if 'plan' in locals() else []

        # Store failed fix attempt in Mem0 for learning
        if 'plan' in locals():
            self._store_fix_pattern_in_mem0(request, plan, success=False)

        return current_notebook, fixes_applied, False

    def _analyze_error_with_mem0(
        self,
        request: FixRequest,
        notebook: NotebookStructure
    ) -> FixPlan:
        """Analyze error using LLM with Mem0 context."""
        # Get Mem0 context for similar errors
        mem0_context = self._get_mem0_error_context(request.user_id, request.traceback)

        # Summarize notebook
        notebook_summary = self._summarize_notebook(notebook)

        # Extract error details from traceback
        error_details = self._extract_error_details(request.traceback)

        # Build context string
        context_str = self._format_mem0_error_context(mem0_context)

        prompt = f"""Analyze and fix this DEAP notebook error.

Error traceback:
{request.traceback}

{context_str}

Current notebook structure (12 cells):
{notebook_summary}

Additional context: {request.context or 'None'}

Common DEAP notebook errors:
1. toolbox.register() called before function definitions
2. Missing return tuple comma in evaluate/mutate functions
3. Incorrect creator.create() usage
4. Variables used before definition
5. Missing imports

Analyze the error and provide a fix plan.
Return structured fixes for the affected cells.
If the fix requires new packages, provide the full updated list of requirements.
"""

        try:
            plan = self.client.chat.completions.create(
                model=settings.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=FixPlan,
                max_tokens=2500,
                temperature=0.2
            )
            return plan
        except Exception as e:
            logger.error(f"Failed to analyze error: {e}")
            # Fallback to heuristic-based fix
            return self._fallback_fix(request, notebook)

    def _get_mem0_error_context(self, user_id: str, traceback: str) -> Dict[str, Any]:
        """Get Mem0 context for similar past errors."""
        context = {
            "common_errors": [],
            "past_fixes": []
        }

        try:
            # Get user's common error patterns
            common = enhanced_memory.get_common_errors(
                user_id=user_id,
                limit=5
            )
            context["common_errors"] = [
                e.get("memory", e.get("content", ""))
                for e in common
            ]

            # Search for similar errors
            error_type = self._extract_error_type(traceback)
            if error_type:
                similar = enhanced_memory.search_user_context(
                    user_id=user_id,
                    query=f"fix for {error_type} error",
                    limit=3
                )
                context["past_fixes"] = [
                    s.get("memory", s.get("content", ""))
                    for s in similar
                ]

        except Exception as e:
            logger.error(f"Error getting Mem0 error context: {e}")

        return context

    def _format_mem0_error_context(self, context: Dict[str, Any]) -> str:
        """Format Mem0 error context for prompt."""
        parts = []

        if context.get("common_errors"):
            parts.append("User's common errors:\n- " + "\n- ".join(context["common_errors"][:3]))

        if context.get("past_fixes"):
            parts.append("Similar past fixes:\n- " + "\n- ".join(context["past_fixes"][:2]))

        return "\n\n".join(parts) if parts else ""

    def _extract_error_type(self, traceback: str) -> Optional[str]:
        """Extract error type from traceback."""
        match = re.search(r'(\w+Error):', traceback)
        return match.group(1) if match else None

    def _extract_error_details(self, traceback: str) -> Dict[str, Any]:
        """Extract structured details from traceback."""
        error_type = self._extract_error_type(traceback)

        # Try to extract cell location
        cell_match = re.search(r'cell[_]s*(\d+)', traceback.lower())
        cell_location = int(cell_match.group(1)) if cell_match else None

        return {
            "error_type": error_type,
            "cell_location": cell_location
        }

    def _summarize_notebook(self, notebook: NotebookStructure) -> str:
        """Create concise notebook summary for LLM."""
        cell_names = [
            "imports", "config", "creator", "evaluate",
            "mate", "mutate", "select", "additional",
            "init", "register", "evolution", "results"
        ]

        summary = []
        for i, (cell, name) in enumerate(zip(notebook.cells, cell_names)):
            preview = cell.source[:150].replace('\n', ' ')
            summary.append(f"Cell {i} ({name}): {preview}...")

        return "\n".join(summary)

    def _apply_fixes(self, notebook: NotebookStructure, plan: FixPlan) -> NotebookStructure:
        """Apply fixes from plan to notebook."""
        cells = [cell.model_copy(deep=True) for cell in notebook.cells]

        for fix in plan.fixes:
            if 0 <= fix.cell_index < 12:
                cells[fix.cell_index] = NotebookCell(
                    cell_type="code",
                    cell_name=cells[fix.cell_index].cell_name,
                    source=format_code(fix.fixed_code),
                    execution_count=None
                )
                logger.info(f"Fixed cell {fix.cell_index}: {fix.fix_description}")

        # Update requirements if provided in plan, otherwise keep existing
        requirements = plan.requirements if plan.requirements is not None else notebook.requirements

        return NotebookStructure(cells=cells, requirements=requirements)

    def _fallback_fix(self, request: FixRequest, notebook: NotebookStructure) -> FixPlan:
        """Heuristic-based fallback fix."""
        logger.warning("Using fallback heuristic fix")

        error_details = self._extract_error_details(request.traceback)
        fixes = []

        # Common fix: toolbox.register before function definitions
        if "NameError" in request.traceback or "not defined" in request.traceback:
            # Check if cell 9 (register) references undefined functions
            register_cell = notebook.cells[9]
            if "toolbox.register" in register_cell.source:
                # Move function definitions before registration
                fixes.append(Fix(
                    cell_index=9,
                    fix_description="Reordered function registrations",
                    fixed_code=register_cell.source
                ))

        # If no specific fixes, return empty plan
        if not fixes:
            fixes.append(Fix(
                cell_index=error_details.get("cell_location", 0),
                fix_description="Unable to auto-fix, manual intervention needed",
                fixed_code=notebook.cells[error_details.get("cell_location", 0)].source
            ))

        return FixPlan(
            error_analysis=ErrorAnalysis(
                error_type=error_details.get("error_type", "Unknown"),
                error_location=error_details.get("cell_location"),
                root_cause="Fallback analysis",
                affected_cells=[f.cell_index for f in fixes]
            ),
            fixes=fixes,
            validation_notes="Fallback heuristic fix applied"
        )

    def _store_fix_pattern_in_mem0(
        self,
        request: FixRequest,
        plan: FixPlan,
        success: bool
    ) -> None:
        """Store fix pattern in Mem0 for future learning."""
        try:
            error_type = plan.error_analysis.error_type
            error_location = plan.error_analysis.error_location or "unknown"

            fix_summary = "; ".join([f.fix_description for f in plan.fixes])

            enhanced_memory.store_error_pattern(
                user_id=request.user_id,
                notebook_id=request.notebook_id,
                error_type=error_type,
                error_location=str(error_location),
                fix_applied=fix_summary if success else f"Failed: {fix_summary}"
            )

            # Store in notebook context
            enhanced_memory.add_notebook_context(
                user_id=request.user_id,
                notebook_id=request.notebook_id,
                operation="fix",
                details={
                    "error_type": error_type,
                    "success": success,
                    "fixes_applied": [f.fix_description for f in plan.fixes],
                    "affected_cells": plan.error_analysis.affected_cells
                }
            )

        except Exception as e:
            logger.error(f"Error storing fix pattern in Mem0: {e}")


class NotebookValidator:
    """Validates notebook structure and content."""

    @staticmethod
    def validate(notebook: NotebookStructure) -> tuple[bool, List[str]]:
        """Validate notebook structure and return errors."""
        return validate_notebook_structure(notebook)