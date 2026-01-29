"""modifier agent - modifies existing notebooks via natural language with mem0 integration."""

import instructor
from groq import Groq
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.config import settings
from app.models import ModifyRequest, NotebookStructure, NotebookCell
from app.memory import enhanced_memory
from app.utils import format_code
from app.prompts.modifier import get_affected_cells_analysis_prompt, get_cell_modification_prompt
import logging
from app.utils.cell_names import CellNameMapper


logger = logging.getLogger(__name__)


class AffectedCellsAnalysis(BaseModel):
    """llm analysis of which cells are affected by a modification."""
    target_cell_index: int = Field(description="Primary cell to modify")
    affected_cells: List[int] = Field(description="Additional cells that need updates")
    reasoning: str = Field(description="Why these cells are affected")


class CellModification(BaseModel):
    """single cell modification."""
    cell_index: int
    new_code: str
    change_description: str


class ModificationResult(BaseModel):
    """result of cell modification."""
    modifications: List[CellModification]
    changes_summary: List[str]
    requirements: Optional[str] = Field(None, description="Updated newline-separated requirements (if changed)")


class NotebookModifier:
    """
    modifies existing deap notebooks using 3-case architecture:
    1. targeted cell modification (when cell_name provided)
    2. notebook-level modification (no cell_name)
    3. error fixing is handled by fixer.py
    """

    # map cell names to indices
    CELL_MAP = CellNameMapper.CELL_MAP

    def __init__(self):
        self.client = instructor.from_groq(
            Groq(api_key=settings.groq_api_key),
            mode=instructor.Mode.JSON
        )

    def modify(
        self,
        request: ModifyRequest
    ) -> tuple[NotebookStructure, List[str], List[int]]:
        """
        modify notebook based on request type.

        returns:
            - modified notebook
            - list of change descriptions
            - list of modified cell indices
        """
        logger.info(f"modifying notebook {request.notebook_id} for user {request.user_id}")
        logger.info(f"instruction: {request.instruction}")
        logger.info(f"cell name: {request.cell_name}")

        if request.cell_name:
            #  targeted cell modification
            return self._targeted_modification(request)
        else:
            #  notebook-level modification
            return self._notebook_level_modification(request)

    # targeted modification

    def _targeted_modification(
        self,
        request: ModifyRequest
    ) -> tuple[NotebookStructure, List[str], List[int]]:
        """
        modify a specific cell with minimal context.

        steps:
        1. get target cell index from cell_name
        2. query mem0 for user preferences and learned dependencies
        3. llm determines affected cells (dynamic dependency analysis)
        4. fetch only required cells
        5. llm modifies those cells
        6. store patterns in mem0
        """
        logger.info(f"targeted modification for cell: {request.cell_name}")

        #  get target cell index
        target_index = self._get_cell_index(request.cell_name)
        if target_index is None:
            logger.error(f"invalid cell name: {request.cell_name}")
            # fallback to notebook-level
            return self._notebook_level_modification(request)

        #  get mem0 context
        mem0_context = self._get_mem0_context_for_cell(
            request.user_id,
            request.notebook_id,
            request.cell_name
        )

        #  llm determines affected cells
        affected_analysis = self._analyze_affected_cells(
            request,
            target_index,
            mem0_context
        )

        all_cell_indices = sorted(set([affected_analysis.target_cell_index] + affected_analysis.affected_cells))
        logger.info(f"cells to modify: {all_cell_indices}")

        #  get cells for modification
        cells_for_llm = [
            {
                "index": i,
                "name": request.notebook.cells[i].cell_name or f"cell_{i}",
                "code": request.notebook.cells[i].source
            }
            for i in all_cell_indices
        ]

        #  modify cells with llm
        modification_result = self._modify_cells_with_llm(
            cells_for_llm,
            request.instruction,
            mem0_context,
            target_cell_name=request.cell_name
        )

        #  apply modifications to notebook
        modified_notebook = self._apply_cell_modifications(
            request.notebook,
            modification_result.modifications
        )
        
        if modification_result.requirements:
            modified_notebook.requirements = modification_result.requirements

        #  store in mem0
        self._store_targeted_modification_in_mem0(
            request,
            target_index,
            all_cell_indices,
            modification_result
        )

        return modified_notebook, modification_result.changes_summary, all_cell_indices

    def _get_cell_index(self, cell_name: str) -> Optional[int]:
        """convert cell name to index."""
        normalized = cell_name.lower().strip()
        return self.CELL_MAP.get(normalized)

    def _get_mem0_context_for_cell(
        self,
        user_id: str,
        notebook_id: str,
        cell_name: str
    ) -> Dict[str, Any]:
        """get relevant mem0 context for cell modification."""
        context = {
            "user_preferences": [],
            "cell_patterns": [],
            "learned_dependencies": [],
            "notebook_history": []
        }

        try:
            # user preferences for this cell type
            prefs = enhanced_memory.search_user_context(
                user_id=user_id,
                query=f"preferences for {cell_name}",
                limit=3
            )
            context["user_preferences"] = [p.get("memory", p.get("content", "")) for p in prefs]

            # past modifications to this cell type
            patterns = enhanced_memory.get_cell_patterns(
                user_id=user_id,
                cell_name=cell_name,
                limit=3
            )
            context["cell_patterns"] = [p.get("memory", p.get("content", "")) for p in patterns]

            # learned dependencies
            deps = enhanced_memory.get_learned_dependencies(
                user_id=user_id,
                cell_name=cell_name
            )
            context["learned_dependencies"] = deps

            # notebook history
            history = enhanced_memory.get_notebook_history(
                user_id=user_id,
                notebook_id=notebook_id,
                limit=3
            )
            context["notebook_history"] = [h.get("memory", h.get("content", "")) for h in history]

        except Exception as e:
            logger.error(f"error getting mem0 context: {e}")

        return context

    def _analyze_affected_cells(
        self,
        request: ModifyRequest,
        target_index: int,
        mem0_context: Dict[str, Any]
    ) -> AffectedCellsAnalysis:
        """use llm to determine which cells are affected."""
        target_cell = request.notebook.cells[target_index]

        # build context string
        context_str = self._format_mem0_context(mem0_context)

        prompt = get_affected_cells_analysis_prompt(
            target_index=target_index,
            target_cell_name=target_cell.cell_name,
            target_cell_source=target_cell.source,
            instruction=request.instruction,
            context_str=context_str
        )

        try:
            analysis = self.client.chat.completions.create(
                model=settings.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=AffectedCellsAnalysis,
                max_tokens=500,
                temperature=0.1
            )
            return analysis
        except Exception as e:
            logger.error(f"failed to analyze affected cells: {e}")
            # fallback: assume only target cell
            return AffectedCellsAnalysis(
                target_cell_index=target_index,
                affected_cells=[],
                reasoning="Fallback: only modifying target cell"
            )

    def _modify_cells_with_llm(
        self,
        cells: List[Dict],
        instruction: str,
        mem0_context: Dict[str, Any],
        target_cell_name: Optional[str] = None
    ) -> ModificationResult:
        """use llm to modify cells."""
        context_str = self._format_mem0_context(mem0_context)

        cells_str = "\n\n".join([
            f"Cell {c['index']} ({c['name']}):\n{c['code']}"
            for c in cells
        ])

        prompt = get_cell_modification_prompt(
            cells_str=cells_str,
            instruction=instruction,
            context_str=context_str
        )

        try:
            result = self.client.chat.completions.create(
                model=settings.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=ModificationResult,
                max_tokens=3000,
                temperature=0.3
            )
            return result
        except Exception as e:
            logger.error(f"failed to modify cells with llm: {e}")
            # fallback: return empty modifications
            return ModificationResult(
                modifications=[],
                changes_summary=[f"Error: {str(e)}"]
            )

    #  notebook-level modification

    def _notebook_level_modification(
        self,
        request: ModifyRequest
    ) -> tuple[NotebookStructure, List[str], List[int]]:
        """
        modify entire notebook for generic/complex changes.

        uses full notebook context + mem0 for holistic modifications.
        """
        logger.info("notebook-level modification (full context)")

        # get mem0 context
        mem0_context = self._get_mem0_context_for_notebook(
            request.user_id,
            request.notebook_id
        )

        # build full notebook context
        all_cells = [
            {
                "index": i,
                "name": cell.cell_name or f"cell_{i}",
                "code": cell.source
            }
            for i, cell in enumerate(request.notebook.cells)
        ]

        # modify with full context
        modification_result = self._modify_cells_with_llm(
            all_cells,
            request.instruction,
            mem0_context,
            target_cell_name=None
        )

        # apply modifications
        modified_notebook = self._apply_cell_modifications(
            request.notebook,
            modification_result.modifications
        )
        
        if modification_result.requirements:
            modified_notebook.requirements = modification_result.requirements

        # extract modified cell indices
        modified_indices = [m.cell_index for m in modification_result.modifications]

        # store in mem0
        self._store_notebook_modification_in_mem0(
            request,
            modified_indices,
            modification_result
        )

        return modified_notebook, modification_result.changes_summary, modified_indices

    def _get_mem0_context_for_notebook(
        self,
        user_id: str,
        notebook_id: str
    ) -> Dict[str, Any]:
        """get mem0 context for notebook-level modifications."""
        context = {
            "user_preferences": [],
            "notebook_history": [],
            "common_patterns": []
        }

        try:
            # overall user preferences
            prefs = enhanced_memory.search_user_context(
                user_id=user_id,
                query="optimization preferences and patterns",
                limit=5
            )
            context["user_preferences"] = [p.get("memory", p.get("content", "")) for p in prefs]

            # notebook history
            history = enhanced_memory.get_notebook_history(
                user_id=user_id,
                notebook_id=notebook_id,
                limit=5
            )
            context["notebook_history"] = [h.get("memory", h.get("content", "")) for h in history]

        except Exception as e:
            logger.error(f"error getting notebook mem0 context: {e}")

        return context

    # helpers

    def _format_mem0_context(self, context: Dict[str, Any]) -> str:
        """format mem0 context for llm prompt."""
        parts = []

        if context.get("user_preferences"):
            parts.append("User preferences:\n- " + "\n- ".join(context["user_preferences"]))

        if context.get("cell_patterns"):
            parts.append("Past cell modifications:\n- " + "\n- ".join(context["cell_patterns"]))

        if context.get("learned_dependencies"):
            deps = ", ".join(context["learned_dependencies"])
            parts.append(f"Learned dependencies: {deps}")

        if context.get("notebook_history"):
            parts.append("Recent notebook changes:\n- " + "\n- ".join(context["notebook_history"][:2]))

        return "\n\n".join(parts) if parts else "No prior context available."

    def _apply_cell_modifications(
        self,
        notebook: NotebookStructure,
        modifications: List[CellModification]
    ) -> NotebookStructure:
        """apply llm modifications to notebook."""
        cells = [cell.model_copy(deep=True) for cell in notebook.cells]

        for mod in modifications:
            if 0 <= mod.cell_index < 12:
                cells[mod.cell_index] = NotebookCell(
                    cell_type="code",
                    cell_name=cells[mod.cell_index].cell_name,
                    source=format_code(mod.new_code),
                    execution_count=None
                )
                logger.info(f"modified cell {mod.cell_index}: {mod.change_description}")

        return NotebookStructure(cells=cells)

    def _store_targeted_modification_in_mem0(
        self,
        request: ModifyRequest,
        target_index: int,
        all_indices: List[int],
        result: ModificationResult
    ) -> None:
        """store targeted modification in mem0."""
        try:
            # store cell modification
            enhanced_memory.store_cell_modification(
                user_id=request.user_id,
                notebook_id=request.notebook_id,
                cell_name=request.cell_name,
                modification_details={
                    "cell_index": target_index,
                    "instruction": request.instruction,
                    "changes": result.changes_summary
                }
            )

            # store dependency pattern if cascading changes occurred
            if len(all_indices) > 1:
                affected = [request.notebook.cells[i].cell_name or f"cell_{i}" for i in all_indices if i != target_index]
                enhanced_memory.store_dependency_pattern(
                    user_id=request.user_id,
                    notebook_id=request.notebook_id,
                    source_cell=request.cell_name,
                    affected_cells=affected,
                    reason=f"Modification: {request.instruction}"
                )

            # store notebook context
            enhanced_memory.add_notebook_context(
                user_id=request.user_id,
                notebook_id=request.notebook_id,
                operation="targeted_modify",
                details={
                    "cell_name": request.cell_name,
                    "instruction": request.instruction,
                    "cells_modified": all_indices,
                    "changes": result.changes_summary
                }
            )

        except Exception as e:
            logger.error(f"error storing in mem0: {e}")

    def _store_notebook_modification_in_mem0(
        self,
        request: ModifyRequest,
        modified_indices: List[int],
        result: ModificationResult
    ) -> None:
        """store notebook-level modification in mem0."""
        try:
            enhanced_memory.add_notebook_context(
                user_id=request.user_id,
                notebook_id=request.notebook_id,
                operation="notebook_modify",
                details={
                    "instruction": request.instruction,
                    "cells_modified": modified_indices,
                    "changes": result.changes_summary
                }
            )
        except Exception as e:
            logger.error(f"error storing in mem0: {e}")