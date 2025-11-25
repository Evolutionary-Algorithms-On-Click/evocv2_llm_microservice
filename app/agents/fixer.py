"""Fixer agent - Fixes broken notebooks based on tracebacks."""

import instructor
from groq import Groq
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.config import settings
from app.models import FixRequest, NotebookStructure, NotebookCell
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


class NotebookFixer:
    """Fixes broken DEAP notebooks based on error tracebacks."""

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
        """Fix notebook with retry loop."""
        logger.info("Fixing notebook based on traceback")

        current_notebook = request.current_notebook
        attempts = 0

        while attempts < max_retries:
            attempts += 1
            logger.info(f"Fix attempt {attempts}/{max_retries}")

            # Analyze error and create fix plan
            plan = self._analyze_error(request, current_notebook)

            # Apply fixes
            fixed_notebook = self._apply_fixes(current_notebook, plan)

            # Validate
            is_valid, errors = validate_notebook_structure(fixed_notebook)

            if is_valid:
                fixes_applied = [f.fix_description for f in plan.fixes]
                return fixed_notebook, fixes_applied, True

            # Update request for retry
            logger.warning(f"Validation failed: {errors}")
            request = FixRequest(
                traceback=f"Validation errors: {'; '.join(errors)}",
                current_notebook=fixed_notebook,
                context=request.context
            )
            current_notebook = fixed_notebook

        # Max retries reached
        logger.error("Failed to fix notebook after max retries")
        fixes_applied = [f"Attempted {attempts} fixes but validation failed"]
        return current_notebook, fixes_applied, False

    def _analyze_error(
        self,
        request: FixRequest,
        notebook: NotebookStructure
    ) -> FixPlan:
        """Analyze error traceback and create fix plan."""
        # Extract cell index from traceback if possible
        error_cell = self._extract_error_cell(request.traceback)

        notebook_code = self._notebook_to_code(notebook)

        prompt = f"""You are an expert at fixing DEAP evolutionary algorithm code.

Error traceback:
{request.traceback}

{f'Additional context: {request.context}' if request.context else ''}

Current notebook code (12 cells):
{notebook_code}

Analyze the error and create a fix plan. The notebook MUST maintain exactly 12 cells:
- Cell 0: imports
- Cell 1: problem config/bounds
- Cell 2: creator.create (FitnessMin/Max, Individual)
- Cell 3: evaluate function (def evaluate)
- Cell 4: mate function (def mate)
- Cell 5: mutate function (def mutate)
- Cell 6: select function (def select)
- Cell 7: additional operators
- Cell 8: initialization (def create_individual)
- Cell 9: toolbox.register() calls (MUST be after all function definitions)
- Cell 10: main evolution loop
- Cell 11: results/plotting

Common DEAP issues:
1. toolbox.register() called before functions are defined → move to cell 9
2. Using class Individual instead of creator.Individual
3. Missing return statement in operators (must return tuple for mutate)
4. Undefined variables/functions
5. Wrong fitness weights

Provide specific fixes for each affected cell."""

        try:
            plan = self.client.chat.completions.create(
                model=settings.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=FixPlan,
                max_tokens=3000,
                temperature=0.2
            )
            return plan
        except Exception as e:
            logger.error(f"Failed to analyze error: {e}")
            return self._fallback_fix_plan(request, error_cell)

    def _extract_error_cell(self, traceback: str) -> Optional[int]:
        """Extract cell index from traceback."""
        # Look for patterns like "in <cell line: 5>" or "Cell In[5]"
        patterns = [
            r'in <cell line:\s*(\d+)>',
            r'Cell In\[(\d+)\]',
            r'<ipython-input-(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, traceback)
            if match:
                cell_num = int(match.group(1))
                # Convert to 0-indexed
                return cell_num - 1 if cell_num > 0 else 0

        return None

    def _notebook_to_code(self, notebook: NotebookStructure) -> str:
        """Convert notebook to code with cell markers."""
        lines = []
        for i, cell in enumerate(notebook.cells):
            lines.append(f"### Cell {i} ###")
            lines.append(cell.source)
            lines.append("")
        return "\n".join(lines)

    def _fallback_fix_plan(
        self,
        request: FixRequest,
        error_cell: Optional[int]
    ) -> FixPlan:
        """Heuristic-based fallback fix plan."""
        traceback_lower = request.traceback.lower()

        fixes = []
        affected_cells = []

        # Common error patterns
        if "nameerror" in traceback_lower and "individual" in traceback_lower:
            # Likely using Individual instead of creator.Individual
            fixes.append(Fix(
                cell_index=2,
                fix_description="Fix Individual class definition in creator",
                fixed_code="""# Create fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)"""
            ))
            affected_cells.append(2)

        if "nameerror" in traceback_lower:
            # Function called before definition
            if error_cell and error_cell == 9:
                fixes.append(Fix(
                    cell_index=9,
                    fix_description="Ensure toolbox.register uses defined functions",
                    fixed_code="""# Register operators in toolbox
toolbox = base.Toolbox()
toolbox.register('individual', create_individual)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', evaluate)
toolbox.register('mate', mate)
toolbox.register('mutate', mutate)
toolbox.register('select', select)"""
                ))
                affected_cells.append(9)

        if "syntaxerror" in traceback_lower and error_cell is not None:
            fixes.append(Fix(
                cell_index=error_cell,
                fix_description=f"Fix syntax error in cell {error_cell}",
                fixed_code="# Syntax error - needs manual review"
            ))
            affected_cells.append(error_cell)

        # Default fix
        if not fixes:
            fixes.append(Fix(
                cell_index=9,
                fix_description="Regenerate toolbox registration",
                fixed_code="""# Register operators in toolbox
toolbox = base.Toolbox()
toolbox.register('individual', create_individual)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', evaluate)
toolbox.register('mate', mate)
toolbox.register('mutate', mutate)
toolbox.register('select', select)"""
            ))
            affected_cells.append(9)

        return FixPlan(
            error_analysis=ErrorAnalysis(
                error_type="Unknown",
                error_location=error_cell,
                root_cause="Automatic detection failed, applying heuristic fixes",
                affected_cells=affected_cells
            ),
            fixes=fixes,
            validation_notes="Applied heuristic fixes based on common DEAP errors"
        )

    def _apply_fixes(self, notebook: NotebookStructure, plan: FixPlan) -> NotebookStructure:
        """Apply fixes to notebook."""
        cells = [cell.model_copy(deep=True) for cell in notebook.cells]

        for fix in plan.fixes:
            if 0 <= fix.cell_index < 12:
                cells[fix.cell_index] = NotebookCell(
                    cell_type="code",
                    source=format_code(fix.fixed_code),
                    execution_count=None
                )

        return NotebookStructure(cells=cells)


class NotebookValidator:
    """Validates notebook structure and code."""

    @staticmethod
    def validate(notebook: NotebookStructure) -> tuple[bool, List[str]]:
        """Comprehensive validation of notebook."""
        return validate_notebook_structure(notebook)

    @staticmethod
    def quick_check(notebook: NotebookStructure) -> bool:
        """Quick structural check."""
        if len(notebook.cells) != 12:
            return False

        # Check critical cells have content
        critical_cells = [0, 2, 3, 9, 10]  # imports, creator, evaluate, toolbox, evolution
        for idx in critical_cells:
            if not notebook.cells[idx].source.strip():
                return False

        return True
