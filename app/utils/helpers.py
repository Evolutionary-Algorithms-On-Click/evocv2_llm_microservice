"""Utility functions for notebook manipulation and validation."""

from typing import Dict, List, Set
import re
from app.models import NotebookStructure, NotebookCell


CELL_ORDER = [
    "imports",
    "problem_config",
    "creator",
    "evaluate_fn",
    "mate_fn",
    "mutate_fn",
    "select_fn",
    "additional_operators",
    "toolbox_register",
    "main_evolution",
    "results"
]


def validate_notebook_structure(notebook: NotebookStructure) -> tuple[bool, List[str]]:
    """Validate that notebook follows the 12-cell structure."""
    errors = []

    if len(notebook.cells) != 12:
        errors.append(f"Expected exactly 12 cells, got {len(notebook.cells)}")
        return False, errors

    # Check that all cells are code cells
    for i, cell in enumerate(notebook.cells, 1):
        if cell.cell_type != "code":
            errors.append(f"Cell {i} must be code type, got {cell.cell_type}")

    # Validate cell 1: imports
    cell_1 = notebook.cells[0].source
    required_imports = ["from deap import", "import numpy", "import random"]
    for imp in required_imports:
        if imp not in cell_1:
            errors.append(f"Cell 1 missing required import: {imp}")

    # Validate cell 3: creator
    cell_3 = notebook.cells[2].source
    if "creator.create" not in cell_3:
        errors.append("Cell 3 must contain creator.create calls")
    if "FitnessMin" not in cell_3 and "FitnessMax" not in cell_3:
        errors.append("Cell 3 must define Fitness (FitnessMin or FitnessMax)")
    if "Individual" not in cell_3:
        errors.append("Cell 3 must define Individual")

    # Validate cell 10: toolbox.register
    cell_10 = notebook.cells[9].source
    if "toolbox.register" not in cell_10:
        errors.append("Cell 10 must contain toolbox.register calls")

    # Validate cell 11: evolution loop
    cell_11 = notebook.cells[10].source
    if "algorithms." not in cell_11 and "for gen in range" not in cell_11:
        errors.append("Cell 11 must contain evolution loop")

    return len(errors) == 0, errors


def extract_function_names(code: str) -> List[str]:
    """Extract function names defined in code."""
    pattern = r'^def\s+(\w+)\s*\('
    return re.findall(pattern, code, re.MULTILINE)


def extract_dependencies(code: str) -> Set[str]:
    """Extract function calls and variable references from code."""
    deps = set()

    # Extract function calls
    func_calls = re.findall(r'(\w+)\s*\(', code)
    deps.update(func_calls)

    # Extract variable references (simple heuristic)
    words = re.findall(r'\b([a-zA-Z_]\w*)\b', code)
    deps.update(words)

    return deps


def analyze_cell_dependencies(notebook: NotebookStructure) -> Dict[int, Set[int]]:
    """Analyze dependencies between cells."""
    cell_definitions = {}
    cell_dependencies = {}

    for i, cell in enumerate(notebook.cells):
        # Track what each cell defines
        func_names = extract_function_names(cell.source)
        cell_definitions[i] = set(func_names)

        # Track what each cell depends on
        deps = extract_dependencies(cell.source)
        cell_dependencies[i] = deps

    # Map dependencies to cell indices
    dependency_graph = {i: set() for i in range(len(notebook.cells))}

    for i, deps in cell_dependencies.items():
        for j, definitions in cell_definitions.items():
            if i != j and deps & definitions:
                dependency_graph[i].add(j)

    return dependency_graph


def get_affected_cells(notebook: NotebookStructure, modified_cell_idx: int) -> List[int]:
    """Get list of cells affected by modification to a specific cell."""
    dep_graph = analyze_cell_dependencies(notebook)

    affected = []
    for cell_idx, dependencies in dep_graph.items():
        if modified_cell_idx in dependencies:
            affected.append(cell_idx)

    # Cell 10 (toolbox.register) almost always needs update
    if modified_cell_idx in [3, 4, 5, 6, 7, 8] and 9 not in affected:
        affected.append(9)

    return sorted(affected)


def create_empty_notebook() -> NotebookStructure:
    """Create an empty 12-cell notebook structure."""
    cells = [
        NotebookCell(cell_type="code", source="", execution_count=None)
        for _ in range(12)
    ]
    return NotebookStructure(cells=cells)


def format_code(code: str) -> str:
    """Basic code formatting."""
    lines = code.split('\n')
    # Remove excessive blank lines
    formatted = []
    prev_blank = False
    for line in lines:
        is_blank = line.strip() == ''
        if is_blank and prev_blank:
            continue
        formatted.append(line)
        prev_blank = is_blank

    return '\n'.join(formatted).strip()


BUILTIN_FUNCTIONS = {
    "sphere": """def evaluate(individual):
    return sum(x**2 for x in individual),""",

    "rastrigin": """def evaluate(individual):
    return 10 * len(individual) + sum(x**2 - 10 * np.cos(2 * np.pi * x) for x in individual),""",

    "rosenbrock": """def evaluate(individual):
    return sum(100 * (individual[i+1] - individual[i]**2)**2 + (1 - individual[i])**2
               for i in range(len(individual) - 1)),""",

    "ackley": """def evaluate(individual):
    n = len(individual)
    sum_sq = sum(x**2 for x in individual)
    sum_cos = sum(np.cos(2 * np.pi * x) for x in individual)
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e,""",

    "griewank": """def evaluate(individual):
    sum_sq = sum(x**2 for x in individual) / 4000
    prod_cos = np.prod([np.cos(individual[i] / np.sqrt(i + 1)) for i in range(len(individual))])
    return sum_sq - prod_cos + 1,"""
}


def get_builtin_function(name: str) -> str:
    """Get builtin objective function by name."""
    return BUILTIN_FUNCTIONS.get(name.lower(), BUILTIN_FUNCTIONS["sphere"])
