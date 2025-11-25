"""Modifier agent - Modifies existing notebooks via natural language."""

import instructor
from groq import Groq
from pydantic import BaseModel, Field
from typing import List, Optional
from app.config import settings
from app.models import ModifyRequest, NotebookStructure, NotebookCell
from app.utils import get_affected_cells, format_code
import logging

logger = logging.getLogger(__name__)


class ModificationIntent(BaseModel):
    """Structured modification intent from LLM."""
    target_cells: List[int] = Field(description="Cell indices to modify (0-11)")
    modification_type: str = Field(description="Type of modification: add, replace, update, delete")
    description: str = Field(description="What changes to make")
    new_code: Optional[str] = Field(None, description="New code if replacing/adding")
    preserve_structure: bool = Field(default=True, description="Keep 12-cell structure")


class ModificationPlan(BaseModel):
    """Complete modification plan."""
    intents: List[ModificationIntent]
    affected_cells: List[int] = Field(description="Cells that need cascading updates")
    changes_summary: List[str]


class NotebookModifier:
    """Modifies existing DEAP notebooks based on natural language."""

    def __init__(self):
        self.client = instructor.from_groq(
            Groq(api_key=settings.groq_api_key),
            mode=instructor.Mode.JSON
        )

    def modify(self, request: ModifyRequest) -> tuple[NotebookStructure, List[str]]:
        """Modify notebook based on natural language instruction."""
        logger.info(f"Modifying notebook: {request.instruction}")

        # Step 1: Analyze instruction and create modification plan
        plan = self._analyze_instruction(request)

        # Step 2: Apply modifications
        modified_notebook = self._apply_modifications(request.current_notebook, plan)

        # Step 3: Update dependent cells
        modified_notebook = self._update_dependent_cells(modified_notebook, plan)

        return modified_notebook, plan.changes_summary

    def _analyze_instruction(self, request: ModifyRequest) -> ModificationPlan:
        """Use LLM to analyze modification instruction."""
        notebook_summary = self._summarize_notebook(request.current_notebook)

        prompt = f"""You are an expert at modifying DEAP evolutionary algorithm code.

Current notebook structure (12 cells):
{notebook_summary}

User instruction: {request.instruction}

Analyze the instruction and create a modification plan. Identify:
1. Which cells need to be modified (by index 0-11)
2. What type of modification (add, replace, update, delete)
3. What specific changes to make
4. Any cascading changes needed

IMPORTANT: The notebook MUST maintain exactly 12 cells in this order:
- Cell 0: imports
- Cell 1: problem config
- Cell 2: creator (FitnessMin/Max, Individual)
- Cell 3: evaluate function
- Cell 4: mate function
- Cell 5: mutate function
- Cell 6: select function
- Cell 7: additional operators
- Cell 8: initialization functions
- Cell 9: toolbox.register() calls
- Cell 10: main evolution loop
- Cell 11: results/plotting

Return a structured modification plan."""

        try:
            plan = self.client.chat.completions.create(
                model=settings.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=ModificationPlan,
                max_tokens=2000
            )
            return plan
        except Exception as e:
            logger.error(f"Failed to analyze instruction: {e}")
            # Fallback: simple heuristic-based modification
            return self._fallback_analysis(request)

    def _summarize_notebook(self, notebook: NotebookStructure) -> str:
        """Create a concise summary of notebook cells."""
        cell_names = [
            "imports", "problem config", "creator", "evaluate",
            "mate", "mutate", "select", "additional ops",
            "initialization", "toolbox.register", "evolution loop", "results"
        ]

        summary = []
        for i, (cell, name) in enumerate(zip(notebook.cells, cell_names)):
            preview = cell.source[:100].replace('\n', ' ')
            summary.append(f"Cell {i} ({name}): {preview}...")

        return "\n".join(summary)

    def _fallback_analysis(self, request: ModifyRequest) -> ModificationPlan:
        """Heuristic-based fallback when LLM fails."""
        instruction_lower = request.instruction.lower()
        intents = []
        changes = []

        # Detect common modification patterns
        if "mutation" in instruction_lower or "mutate" in instruction_lower:
            intents.append(ModificationIntent(
                target_cells=[5],
                modification_type="update",
                description="Modify mutation operator",
                preserve_structure=True
            ))
            changes.append("Updated mutation operator")

        if "crossover" in instruction_lower or "mate" in instruction_lower:
            intents.append(ModificationIntent(
                target_cells=[4],
                modification_type="update",
                description="Modify crossover operator",
                preserve_structure=True
            ))
            changes.append("Updated crossover operator")

        if "selection" in instruction_lower or "select" in instruction_lower:
            intents.append(ModificationIntent(
                target_cells=[6],
                modification_type="update",
                description="Modify selection operator",
                preserve_structure=True
            ))
            changes.append("Updated selection operator")

        if "population" in instruction_lower or "generations" in instruction_lower:
            intents.append(ModificationIntent(
                target_cells=[10],
                modification_type="update",
                description="Modify algorithm parameters",
                preserve_structure=True
            ))
            changes.append("Updated algorithm parameters")

        if "plot" in instruction_lower or "visualization" in instruction_lower:
            intents.append(ModificationIntent(
                target_cells=[11],
                modification_type="update",
                description="Modify plotting code",
                preserve_structure=True
            ))
            changes.append("Updated plotting")

        # Default: modify evolution loop
        if not intents:
            intents.append(ModificationIntent(
                target_cells=[10],
                modification_type="update",
                description="General modification to evolution logic",
                preserve_structure=True
            ))
            changes.append("Modified evolution loop")

        affected = list(set(intent.target_cells[0] for intent in intents))
        if 9 not in affected and any(c in [3, 4, 5, 6, 7, 8] for c in affected):
            affected.append(9)  # toolbox.register likely needs update

        return ModificationPlan(
            intents=intents,
            affected_cells=affected,
            changes_summary=changes
        )

    def _apply_modifications(
        self,
        notebook: NotebookStructure,
        plan: ModificationPlan
    ) -> NotebookStructure:
        """Apply planned modifications to notebook."""
        cells = [cell.model_copy(deep=True) for cell in notebook.cells]

        for intent in plan.intents:
            for cell_idx in intent.target_cells:
                if 0 <= cell_idx < 12:
                    cells[cell_idx] = self._modify_cell(
                        cells[cell_idx],
                        intent,
                        cell_idx
                    )

        return NotebookStructure(cells=cells)

    def _modify_cell(
        self,
        cell: NotebookCell,
        intent: ModificationIntent,
        cell_idx: int
    ) -> NotebookCell:
        """Modify a specific cell based on intent."""
        if intent.new_code:
            # Direct replacement
            return NotebookCell(
                cell_type="code",
                source=format_code(intent.new_code),
                execution_count=None
            )

        # Generate modification using LLM
        prompt = f"""Modify this DEAP code cell based on the instruction.

Current code:
{cell.source}

Modification: {intent.description}

Return ONLY the modified Python code, no explanations or markdown."""

        try:
            response = self.client.chat.completions.create(
                model=settings.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=None,
                max_tokens=1500,
                temperature=0.3
            )

            modified_code = response.choices[0].message.content.strip()
            # Clean up any markdown code blocks
            if modified_code.startswith("```"):
                lines = modified_code.split('\n')
                modified_code = '\n'.join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            return NotebookCell(
                cell_type="code",
                source=format_code(modified_code),
                execution_count=None
            )
        except Exception as e:
            logger.error(f"Failed to modify cell {cell_idx}: {e}")
            return cell  # Return unchanged

    def _update_dependent_cells(
        self,
        notebook: NotebookStructure,
        plan: ModificationPlan
    ) -> NotebookStructure:
        """Update cells that depend on modified cells."""
        cells = [cell.model_copy(deep=True) for cell in notebook.cells]

        # Always regenerate toolbox.register (cell 9) if operators changed
        operator_cells = [3, 4, 5, 6, 7, 8]
        if any(c in plan.affected_cells for c in operator_cells):
            cells[9] = self._regenerate_toolbox_register(cells)

        return NotebookStructure(cells=cells)

    def _regenerate_toolbox_register(self, cells: List[NotebookCell]) -> NotebookCell:
        """Regenerate toolbox.register cell based on current operators."""
        # Extract function names from operator cells
        eval_func = "evaluate" if "def evaluate" in cells[3].source else None
        mate_func = "mate" if "def mate" in cells[4].source else None
        mutate_func = "mutate" if "def mutate" in cells[5].source else None
        select_func = "select" if "def select" in cells[6].source else None
        init_func = "create_individual" if "def create_individual" in cells[8].source else None

        source = """# Register operators in toolbox
toolbox = base.Toolbox()"""

        if init_func:
            source += f"\ntoolbox.register('individual', {init_func})"
            source += "\ntoolbox.register('population', tools.initRepeat, list, toolbox.individual)"

        if eval_func:
            source += f"\ntoolbox.register('evaluate', {eval_func})"

        if mate_func:
            source += f"\ntoolbox.register('mate', {mate_func})"

        if mutate_func:
            source += f"\ntoolbox.register('mutate', {mutate_func})"

        if select_func:
            source += f"\ntoolbox.register('select', {select_func})"

        return NotebookCell(
            cell_type="code",
            source=format_code(source),
            execution_count=None
        )
