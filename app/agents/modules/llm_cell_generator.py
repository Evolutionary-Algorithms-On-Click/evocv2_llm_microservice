"""LLM-based cell generation for DEAP notebooks."""

from typing import Dict, Any, Optional
import instructor
from groq import Groq
from pydantic import BaseModel, Field
import logging

from app.config import settings
from app.agents.modules.cell_names import CellNameMapper

logger = logging.getLogger(__name__)


class CellGenerationResult(BaseModel):
    """Structured result from LLM cell generation."""
    source_code: str = Field(..., description="The Python source code for this cell")
    explanation: str = Field(..., description="Brief explanation of what this cell does")


class SingleCellCode(BaseModel):
    """A single cell's code in the complete notebook."""
    cell_name: str = Field(..., description="Name of the cell (e.g., 'imports', 'config', 'creator', etc.)")
    source_code: str = Field(..., description="The Python source code for this cell")


class CompleteNotebookGeneration(BaseModel):
    """Complete notebook with all 12 cells generated in a single pass."""
    cells: list[SingleCellCode] = Field(..., min_length=12, max_length=12, description="All 12 cells in order")


class LLMCellGenerator:
    """Generates individual notebook cells using LLM."""

    def __init__(self):
        self.client = instructor.from_groq(
            Groq(api_key=settings.groq_api_key),
            mode=instructor.Mode.JSON,
            model = settings.model_name
        )
        self.cell_mapper = CellNameMapper()

    def generate_all_cells(self, problem_data: Dict[str, Any]) -> CompleteNotebookGeneration:
        """
        Generate all 12 cells in a single LLM pass.

        Args:
            problem_data: Structured problem data from RequestParser

        Returns:
            CompleteNotebookGeneration with all 12 cells
        """
        logger.info("Generating all 12 cells in a single LLM pass")

        # Build comprehensive prompt for all cells
        prompt = self._build_complete_notebook_prompt(problem_data)

        try:
            result: CompleteNotebookGeneration = self.client.chat.completions.create(
                model=settings.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert in DEAP (Distributed Evolutionary Algorithms in Python).
Generate a complete, consistent 12-cell DEAP notebook.

CRITICAL REQUIREMENTS:
1. Generate ALL imports ONLY in the 'imports' cell (cell 0)
2. Do NOT repeat imports in other cells
3. Use creator.create() ONLY ONCE in the 'creator' cell (cell 2) - never redefine fitness or individual classes
4. Ensure all cells work together as a cohesive notebook
5. Reference variables and functions defined in previous cells
6. Generate clean, efficient, executable Python code"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_model=CompleteNotebookGeneration,
                temperature=0.3,
                max_tokens=8000
            )

            logger.info(f"Successfully generated all 12 cells in single pass")
            return result

        except Exception as e:
            logger.error(f"LLM complete notebook generation failed: {e}")
            # Fallback to basic templates
            return self._fallback_complete_notebook(problem_data)

    def generate_cell(
        self,
        cell_index: int,
        problem_data: Dict[str, Any],
        context: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate a single cell using LLM.

        Args:
            cell_index: Index of the cell to generate (0-11)
            problem_data: Structured problem data from RequestParser
            context: Optional context from previously generated cells

        Returns:
            Source code for the cell
        """
        cell_name = self.cell_mapper.get_cell_name(cell_index)
        cell_description = self.cell_mapper.get_cell_description(cell_name)

        logger.info(f"Generating cell {cell_index} ({cell_name}) using LLM")

        # Build context-aware prompt
        prompt = self._build_prompt(cell_index, cell_name, cell_description, problem_data, context)

        try:
            result: CellGenerationResult = self.client.chat.completions.create(
                model=settings.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in DEAP (Distributed Evolutionary Algorithms in Python). Generate clean, efficient, and correct DEAP code for evolutionary algorithms."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_model=CellGenerationResult,
                temperature=0.3,
                max_tokens=2000
            )

            logger.info(f"Successfully generated cell {cell_index}: {result.explanation}")
            return result.source_code

        except Exception as e:
            logger.error(f"LLM cell generation failed for cell {cell_index}: {e}")
            # Fallback to basic template
            return self._fallback_template(cell_index, problem_data)

    def _build_prompt(
        self,
        cell_index: int,
        cell_name: str,
        cell_description: str,
        problem_data: Dict[str, Any],
        context: Optional[Dict[str, str]] = None
    ) -> str:
        """Build a detailed prompt for cell generation."""

        prompt = f"""Generate Python code for a DEAP evolutionary algorithm notebook cell.

**Cell Information:**
- Cell Index: {cell_index}
- Cell Name: {cell_name}
- Purpose: {cell_description}

**Problem Specification:**
- Problem Name: {problem_data.get('problem_name', 'Optimization Problem')}
- Goal: {problem_data.get('goal_description', 'Optimize the objective function')}
- Objective Type: {problem_data.get('objective_type', 'minimization')}
- Objective Function: {problem_data.get('objective_function', '')}
"""

        if problem_data.get('formal_equation'):
            prompt += f"- Formal Equation: {problem_data['formal_equation']}\n"

        prompt += f"""
**Solution Representation:**
- Type: {problem_data.get('solution_representation', 'real-valued')}
- Size: {problem_data.get('solution_size', 10)} variables
- Domain: [{problem_data.get('lower_bounds', [0])[0]}, {problem_data.get('upper_bounds', [1])[0]}]
"""

        # Add cell-specific instructions
        prompt += self._get_cell_specific_instructions(cell_index, problem_data, context)

        prompt += """
**Requirements:**
1. Generate ONLY the Python code, no markdown or explanations in the code
2. Use proper DEAP conventions and best practices
3. Ensure code is executable and syntactically correct
4. Include necessary comments for clarity
5. Follow PEP 8 style guidelines

Provide the source code that should go in this cell."""

        return prompt

    def _get_cell_specific_instructions(
        self,
        cell_index: int,
        problem_data: Dict[str, Any],
        context: Optional[Dict[str, str]] = None
    ) -> str:
        """Get cell-specific generation instructions."""

        if cell_index == 0:  # Imports
            instructions = """
**Cell-Specific Instructions:**
Generate import statements for:
- deap (base, creator, tools, algorithms)
- numpy as np
- random
"""
            if problem_data.get('output_visualization'):
                instructions += "- matplotlib.pyplot as plt\n"
            return instructions

        elif cell_index == 1:  # Config
            return f"""
**Cell-Specific Instructions:**
Define problem configuration:
- DIMENSIONS = {problem_data.get('solution_size', 10)}
- LOWER_BOUND = {problem_data.get('lower_bounds', [0])}
- UPPER_BOUND = {problem_data.get('upper_bounds', [1])}
- Set random seeds for reproducibility (random.seed(42), np.random.seed(42))
"""

        elif cell_index == 2:  # Creator
            obj_type = problem_data.get('objective_type', 'minimization')
            weights = "(-1.0,)" if "min" in obj_type.lower() else "(1.0,)"
            return f"""
**Cell-Specific Instructions:**
Create DEAP fitness and individual classes:
- Use creator.create() to define FitnessMin or FitnessMax with weights={weights}
- Use creator.create() to define Individual as a list with the fitness class
"""

        elif cell_index == 3:  # Evaluate
            return f"""
**Cell-Specific Instructions:**
Define the evaluation function 'evaluate(individual)':
- Objective: {problem_data.get('objective_function', 'minimize sum of squares')}
- {problem_data.get('fitness_description', '')}
- Return a tuple (fitness_value,) - note the comma for single objective
- Implement the fitness calculation based on the problem description
"""

        elif cell_index == 4:  # Crossover
            cx_op = problem_data.get('crossover_operator', 'blend')
            return f"""
**Cell-Specific Instructions:**
Define the crossover function 'mate(ind1, ind2)':
- Use DEAP's {cx_op} crossover or tools.cxBlend
- Return (ind1, ind2) after crossover
- Crossover probability will be handled in the evolution loop
"""

        elif cell_index == 5:  # Mutation
            mut_op = problem_data.get('mutation_operator', 'gaussian')
            return f"""
**Cell-Specific Instructions:**
Define the mutation function 'mutate(individual)':
- Use DEAP's {mut_op} mutation or tools.mutGaussian
- Consider the bounds: LOWER_BOUND and UPPER_BOUND
- Return (individual,) - note the comma for DEAP compatibility
"""

        elif cell_index == 6:  # Selection
            sel_method = problem_data.get('selection_method', 'tournament')
            return f"""
**Cell-Specific Instructions:**
Define the selection function 'select(individuals, k)':
- Use DEAP's {sel_method} selection (e.g., tools.selTournament)
- Return k selected individuals
"""

        elif cell_index == 7:  # Additional operators
            custom_ops = problem_data.get('custom_operators', '')
            if custom_ops:
                return f"""
**Cell-Specific Instructions:**
Define additional custom operators:
{custom_ops}
"""
            return """
**Cell-Specific Instructions:**
Add a comment that additional custom operators can be defined here if needed.
"""

        elif cell_index == 8:  # Initialization
            return """
**Cell-Specific Instructions:**
Define the initialization function 'create_individual()':
- Create and return a creator.Individual
- Initialize with random values within LOWER_BOUND and UPPER_BOUND
- Use list comprehension with random.uniform()
"""

        elif cell_index == 9:  # Toolbox registration
            return """
**Cell-Specific Instructions:**
Create and register operators in the DEAP toolbox:
- Create toolbox = base.Toolbox()
- Register: individual, population, evaluate, mate, mutate, select
- Use toolbox.register() for each operator
"""

        elif cell_index == 10:  # Evolution loop
            return f"""
**Cell-Specific Instructions:**
Implement the main evolutionary algorithm:
- Initialize population with size {problem_data.get('population_size', 100)}
- Create statistics (tools.Statistics) to track min, max, avg, std
- Create Hall of Fame (tools.HallOfFame) if requested
- Run evolution for {problem_data.get('num_generations', 50)} generations
- Use algorithms.eaSimple or implement custom loop
- Crossover probability: {problem_data.get('crossover_probability', 0.7)}
- Mutation probability: {problem_data.get('mutation_probability', 0.2)}
"""

        elif cell_index == 11:  # Results and plots
            show_viz = problem_data.get('output_visualization', False)
            return f"""
**Cell-Specific Instructions:**
Display results and create visualizations:
- Print best individuals from Hall of Fame
- Print final statistics (min, avg, max, std)
{"- Create matplotlib plot showing fitness evolution over generations" if show_viz else ""}
- Use plt.figure(), plt.plot(), plt.legend(), plt.show()
"""

        return ""

    def _build_complete_notebook_prompt(self, problem_data: Dict[str, Any]) -> str:
        """Build a comprehensive prompt for generating all 12 cells at once."""

        # Extract other_specifications if present
        other_specs = problem_data.get('other_specifications', {})
        other_specs_text = ""
        if other_specs:
            other_specs_text = "\n**Additional Specifications:**\n"
            for key, value in other_specs.items():
                other_specs_text += f"- {key}: {value}\n"

        prompt = f"""Generate a complete 12-cell DEAP evolutionary algorithm notebook.

**Problem Specification:**
- Problem Name: {problem_data.get('problem_name', 'Optimization Problem')}
- Goal: {problem_data.get('goal_description', 'Optimize the objective function')}
- Objective Type: {problem_data.get('objective_type', 'minimization')}
- Objective Function: {problem_data.get('objective_function', '')}
"""

        if problem_data.get('formal_equation'):
            prompt += f"- Formal Equation: {problem_data['formal_equation']}\n"

        prompt += f"""
**Solution Representation:**
- Type: {problem_data.get('solution_representation', 'real-valued')}
- Size: {problem_data.get('solution_size', 10)} variables
- Domain: [{problem_data.get('lower_bounds', [0])[0]}, {problem_data.get('upper_bounds', [1])[0]}]

**Genetic Operators:**
- Selection: {problem_data.get('selection_method', 'tournament')}
- Crossover: {problem_data.get('crossover_operator', 'blend')} (probability: {problem_data.get('crossover_probability', 0.7)})
- Mutation: {problem_data.get('mutation_operator', 'gaussian')} (probability: {problem_data.get('mutation_probability', 0.2)})

**Algorithm Parameters:**
- Population Size: {problem_data.get('population_size', 100)}
- Generations: {problem_data.get('num_generations', 50)}
- Visualization: {problem_data.get('output_visualization', False)}
{other_specs_text}

**12-Cell Structure:**

Cell 0 (imports): Import ALL required libraries (deap, numpy, random, matplotlib if needed)
Cell 1 (config): Define configuration constants (DIMENSIONS, LOWER_BOUND, UPPER_BOUND, seeds)
Cell 2 (creator): Create fitness and individual classes using creator.create() - ONLY HERE
Cell 3 (evaluate): Define evaluate(individual) function implementing the objective
Cell 4 (crossover): Define mate(ind1, ind2) function for crossover
Cell 5 (mutation): Define mutate(individual) function for mutation
Cell 6 (selection): Define select(individuals, k) function for selection
Cell 7 (additional_operators): Additional custom operators or comments
Cell 8 (initialization): Define create_individual() function for initialization
Cell 9 (toolbox_registration): Create toolbox and register all operators
Cell 10 (evolution_loop): Initialize population, run evolution with statistics
Cell 11 (results_and_plots): Display results, print statistics, create plots if requested

**CRITICAL RULES:**
1. Cell 0 must contain ALL imports - no imports in other cells
2. Cell 2 is the ONLY cell that calls creator.create() - never repeat this
3. Each cell should reference previous cells' definitions (no redefinitions)
4. Code must be consistent across all cells
5. All cells together form a complete, executable notebook

Generate the complete notebook as a JSON array of 12 cells."""

        return prompt

    def _fallback_complete_notebook(self, problem_data: Dict[str, Any]) -> CompleteNotebookGeneration:
        """Generate fallback complete notebook using templates."""

        cell_names = [
            "imports", "config", "creator", "evaluate", "crossover",
            "mutation", "selection", "additional_operators", "initialization",
            "toolbox_registration", "evolution_loop", "results_and_plots"
        ]

        cells = []
        for i, name in enumerate(cell_names):
            source = self._fallback_template(i, problem_data)
            cells.append(SingleCellCode(cell_name=name, source_code=source))

        return CompleteNotebookGeneration(cells=cells)

    def _fallback_template(self, cell_index: int, problem_data: Dict[str, Any]) -> str:
        """Fallback templates when LLM fails."""

        templates = {
            0: """from deap import base, creator, tools, algorithms
import numpy as np
import random""",
            1: f"""# Problem configuration
DIMENSIONS = {problem_data.get('solution_size', 10)}
LOWER_BOUND = {problem_data.get('lower_bounds', [0])}
UPPER_BOUND = {problem_data.get('upper_bounds', [1])}

random.seed(42)
np.random.seed(42)""",
            2: """# Create fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)""",
            3: """def evaluate(individual):
    return sum(x**2 for x in individual),""",
            4: """def mate(ind1, ind2):
    tools.cxBlend(ind1, ind2, 0.5)
    return ind1, ind2""",
            5: """def mutate(individual):
    tools.mutGaussian(individual, mu=0, sigma=1, indpb=0.2)
    return individual,""",
            6: """def select(individuals, k):
    return tools.selTournament(individuals, k, tournsize=3)""",
            7: """# Additional custom operators can be defined here""",
            8: """def create_individual():
    return creator.Individual([
        random.uniform(LOWER_BOUND[i], UPPER_BOUND[i])
        for i in range(DIMENSIONS)
    ])""",
            9: """toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", mate)
toolbox.register("mutate", mutate)
toolbox.register("select", select)""",
            10: f"""population = toolbox.population(n={problem_data.get('population_size', 100)})

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

hof = tools.HallOfFame(10)

population, logbook = algorithms.eaSimple(
    population, toolbox,
    cxpb={problem_data.get('crossover_probability', 0.7)},
    mutpb={problem_data.get('mutation_probability', 0.2)},
    ngen={problem_data.get('num_generations', 50)},
    stats=stats,
    halloffame=hof,
    verbose=True
)""",
            11: """print("\\nBest individuals:")
for i, ind in enumerate(hof, 1):
    print(f"{i}. Fitness: {ind.fitness.values[0]:.6f}")

print("\\nFinal Statistics:")
record = logbook[-1]
print(f"Min: {record['min']:.6f}")
print(f"Avg: {record['avg']:.6f}")"""
        }

        return templates.get(cell_index, f"# Cell {cell_index}")
