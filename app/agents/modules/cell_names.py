"""Cell naming and type definitions for 12-cell DEAP notebooks."""

from typing import Dict, List


class CellNameMapper:
    """Maps cell indices to descriptive names and provides metadata."""

    # Standard 12-cell structure
    CELL_NAMES = {
        0: "imports",
        1: "config",
        2: "creator",
        3: "evaluate",
        4: "crossover",
        5: "mutation",
        6: "selection",
        7: "additional_operators",
        8: "initialization",
        9: "toolbox_registration",
        10: "evolution_loop",
        11: "results_and_plots"
    }

    CELL_DESCRIPTIONS = {
        "imports": "Import all required libraries (DEAP, NumPy, matplotlib, etc.)",
        "config": "Problem configuration (dimensions, bounds, random seed)",
        "creator": "Create fitness and individual classes using creator.create",
        "evaluate": "Define the objective/fitness evaluation function",
        "crossover": "Define the crossover/mating function",
        "mutation": "Define the mutation function",
        "selection": "Define the selection function",
        "additional_operators": "Define any additional custom operators",
        "initialization": "Define individual initialization function",
        "toolbox_registration": "Register all operators in the DEAP toolbox",
        "evolution_loop": "Main evolutionary algorithm loop",
        "results_and_plots": "Display results, best solutions, and generate plots"
    }

    @classmethod
    def get_cell_name(cls, index: int) -> str:
        """Get descriptive name for cell at given index."""
        return cls.CELL_NAMES.get(index, f"cell_{index}")

    @classmethod
    def get_cell_description(cls, cell_name: str) -> str:
        """Get description for a cell by name."""
        return cls.CELL_DESCRIPTIONS.get(cell_name, "")

    @classmethod
    def get_index_by_name(cls, cell_name: str) -> int:
        """Get cell index by name."""
        for idx, name in cls.CELL_NAMES.items():
            if name == cell_name:
                return idx
        return -1

    @classmethod
    def get_all_cell_names(cls) -> List[str]:
        """Get list of all cell names in order."""
        return [cls.CELL_NAMES[i] for i in range(12)]

    @classmethod
    def get_cell_metadata(cls, index: int) -> Dict[str, str]:
        """Get metadata for a cell."""
        name = cls.get_cell_name(index)
        return {
            "index": index,
            "name": name,
            "description": cls.CELL_DESCRIPTIONS.get(name, "")
        }


# Dependency mapping: which cells depend on which other cells
# BETA (yet to test)
CELL_DEPENDENCIES = {
    "toolbox_registration": ["evaluate", "crossover", "mutation", "selection", "initialization"],
    "evolution_loop": ["toolbox_registration", "config"],
    "results_and_plots": ["evolution_loop"]
}


def get_dependent_cells(cell_name: str) -> List[str]:
    """Get list of cells that the given cell depends on."""
    return CELL_DEPENDENCIES.get(cell_name, [])


def get_cells_dependent_on(cell_name: str) -> List[str]:
    """Get list of cells that depend on the given cell."""
    dependents = []
    for cell, deps in CELL_DEPENDENCIES.items():
        if cell_name in deps:
            dependents.append(cell)
    return dependents
