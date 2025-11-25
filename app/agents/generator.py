"""Generator agent - Creates 12-cell DEAP notebooks using LLM (New version)."""

import logging
from typing import Dict, Any

from app.models import GenerateRequest, NotebookStructure, NotebookCell
from app.agents.modules.request_parser import RequestParser
from app.agents.modules.cell_names import CellNameMapper
from app.agents.modules.llm_cell_generator import LLMCellGenerator

logger = logging.getLogger(__name__)


class NotebookGenerator:
    """Generates complete 12-cell DEAP notebooks using LLM."""

    def __init__(self):
        self.request_parser = RequestParser()
        self.cell_mapper = CellNameMapper()
        self.llm_generator = LLMCellGenerator()

    def generate(self, request: GenerateRequest) -> NotebookStructure:
        """Generate a complete 12-cell notebook from specification."""
        logger.info(f"Generating notebook for session {request.session_id}")

        # Parse the flexible request format into structured data
        problem_data = self.request_parser.extract_structured_data(request)

        logger.info(f"Parsed problem data: {problem_data.get('problem_name')}")

        # Generate all cells in a single LLM pass
        complete_notebook = self.llm_generator.generate_all_cells(problem_data)

        # Convert LLM response to NotebookCell objects
        cells = []
        for cell_index, cell_data in enumerate(complete_notebook.cells):
            cell = NotebookCell(
                cell_type="code",
                cell_name=cell_data.cell_name,
                source=cell_data.source_code,
                execution_count=None,
                metadata={"cell_index": cell_index}
            )
            cells.append(cell)
            logger.info(f"Processed cell {cell_index}: {cell_data.cell_name}")

        notebook = NotebookStructure(cells=cells)
        logger.info(f"Successfully generated complete notebook for session {request.session_id}")

        return notebook
