"""Generator agent - Creates 12-cell DEAP notebooks from specifications."""

import instructor
from groq import Groq
from app.config import settings
from app.models import (
    GenerateRequest, NotebookStructure, NotebookCell,
    ProblemConfig, AlgorithmConfig, OperatorConfig, Features
)
from app.utils import get_builtin_function, format_code
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class NotebookGenerator:
    """Generates complete 12-cell DEAP notebooks."""

    def __init__(self):
        self.client = instructor.from_groq(
            Groq(api_key=settings.groq_api_key),
            mode=instructor.Mode.JSON
        )

    def generate(self, request: GenerateRequest) -> NotebookStructure:
        """Generate a complete 12-cell notebook from specification."""
        logger.info(f"Generating notebook for session {request.session_id}")

        # Build cells systematically
        cells = []

        # Cell 1: Imports
        cells.append(self._generate_cell_1(request))

        # Cell 2: Problem config
        cells.append(self._generate_cell_2(request))

        # Cell 3: Creator
        cells.append(self._generate_cell_3(request))

        # Cell 4: Evaluate function
        cells.append(self._generate_cell_4(request))

        # Cell 5: Mate function
        cells.append(self._generate_cell_5(request))

        # Cell 6: Mutate function
        cells.append(self._generate_cell_6(request))

        # Cell 7: Select function
        cells.append(self._generate_cell_7(request))

        # Cell 8: Additional operators (if needed)
        cells.append(self._generate_cell_8(request))

        # Cell 9: Constraint/initialization functions
        cells.append(self._generate_cell_9(request))

        # Cell 10: Toolbox registration
        cells.append(self._generate_cell_10(request))

        # Cell 11: Main evolution loop
        cells.append(self._generate_cell_11(request))

        # Cell 12: Results and plotting
        cells.append(self._generate_cell_12(request))

        return NotebookStructure(cells=cells)

    def _generate_cell_1(self, request: GenerateRequest) -> NotebookCell:
        """Cell 1: Imports."""
        imports = """from deap import base, creator, tools, algorithms
import numpy as np
import random"""

        if request.features.plotting:
            imports += "\nimport matplotlib.pyplot as plt"

        if request.features.checkpoint:
            imports += "\nimport pickle"

        if request.features.parallel:
            imports += "\nfrom multiprocessing import Pool"

        return NotebookCell(
            cell_type="code",
            source=format_code(imports),
            execution_count=None
        )

    def _generate_cell_2(self, request: GenerateRequest) -> NotebookCell:
        """Cell 2: Problem configuration."""
        prob = request.problem
        source = f"""# Problem configuration
DIMENSIONS = {prob.dimensions}
LOWER_BOUND = {prob.bounds.lower}
UPPER_BOUND = {prob.bounds.upper}

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
"""
        return NotebookCell(cell_type="code", source=format_code(source))

    def _generate_cell_3(self, request: GenerateRequest) -> NotebookCell:
        """Cell 3: Creator setup."""
        minimize = request.problem.objective.minimize
        fitness_type = "FitnessMin" if minimize else "FitnessMax"
        weights = "(-1.0,)" if minimize else "(1.0,)"

        source = f"""# Create fitness and individual classes
creator.create("{fitness_type}", base.Fitness, weights={weights})
creator.create("Individual", list, fitness=creator.{fitness_type})
"""
        return NotebookCell(cell_type="code", source=format_code(source))

    def _generate_cell_4(self, request: GenerateRequest) -> NotebookCell:
        """Cell 4: Evaluate function."""
        obj = request.problem.objective

        if obj.type == "builtin" and obj.name:
            source = get_builtin_function(obj.name)
        elif obj.type == "custom" and obj.code:
            source = obj.code
        else:
            # Default to sphere
            source = get_builtin_function("sphere")

        return NotebookCell(cell_type="code", source=format_code(source))

    def _generate_cell_5(self, request: GenerateRequest) -> NotebookCell:
        """Cell 5: Crossover function."""
        cx_op = request.operators.crossover
        params = request.operators.crossover_params

        if cx_op == "cxBlend":
            alpha = params.get("alpha", 0.5)
            source = f"""def mate(ind1, ind2):
    tools.cxBlend(ind1, ind2, {alpha})
    return ind1, ind2"""
        elif cx_op == "cxSimulatedBinary":
            eta = params.get("eta", 20.0)
            source = f"""def mate(ind1, ind2):
    tools.cxSimulatedBinary(ind1, ind2, {eta})
    return ind1, ind2"""
        else:
            # Default uniform crossover
            indpb = params.get("indpb", 0.5)
            source = f"""def mate(ind1, ind2):
    tools.cxUniform(ind1, ind2, {indpb})
    return ind1, ind2"""

        return NotebookCell(cell_type="code", source=format_code(source))

    def _generate_cell_6(self, request: GenerateRequest) -> NotebookCell:
        """Cell 6: Mutation function."""
        mut_op = request.operators.mutation
        params = request.operators.mutation_params

        if mut_op == "mutGaussian":
            mu = params.get("mu", 0)
            sigma = params.get("sigma", 1)
            indpb = params.get("indpb", 0.2)
            source = f"""def mutate(individual):
    tools.mutGaussian(individual, mu={mu}, sigma={sigma}, indpb={indpb})
    return individual,"""
        elif mut_op == "mutPolynomialBounded":
            eta = params.get("eta", 20.0)
            source = f"""def mutate(individual):
    tools.mutPolynomialBounded(individual, eta={eta}, low=LOWER_BOUND, up=UPPER_BOUND, indpb=0.2)
    return individual,"""
        else:
            # Default
            source = f"""def mutate(individual):
    tools.mutGaussian(individual, mu=0, sigma=1, indpb=0.2)
    return individual,"""

        return NotebookCell(cell_type="code", source=format_code(source))

    def _generate_cell_7(self, request: GenerateRequest) -> NotebookCell:
        """Cell 7: Selection function."""
        sel_op = request.operators.selection
        params = request.operators.selection_params

        if sel_op == "selTournament":
            tournsize = params.get("tournsize", 3)
            source = f"""def select(individuals, k):
    return tools.selTournament(individuals, k, tournsize={tournsize})"""
        elif sel_op == "selRoulette":
            source = """def select(individuals, k):
    return tools.selRoulette(individuals, k)"""
        elif sel_op == "selBest":
            source = """def select(individuals, k):
    return tools.selBest(individuals, k)"""
        else:
            source = f"""def select(individuals, k):
    return tools.selTournament(individuals, k, tournsize=3)"""

        return NotebookCell(cell_type="code", source=format_code(source))

    def _generate_cell_8(self, request: GenerateRequest) -> NotebookCell:
        """Cell 8: Additional operators (placeholder)."""
        source = "# Additional operators can be defined here if needed"
        return NotebookCell(cell_type="code", source=source)

    def _generate_cell_9(self, request: GenerateRequest) -> NotebookCell:
        """Cell 9: Individual initialization."""
        prob = request.problem
        source = f"""def create_individual():
    return creator.Individual([
        random.uniform(LOWER_BOUND[i], UPPER_BOUND[i])
        for i in range(DIMENSIONS)
    ])"""
        return NotebookCell(cell_type="code", source=format_code(source))

    def _generate_cell_10(self, request: GenerateRequest) -> NotebookCell:
        """Cell 10: Toolbox registration."""
        source = """# Register operators in toolbox
toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", mate)
toolbox.register("mutate", mutate)
toolbox.register("select", select)"""

        if request.features.parallel:
            source += "\n\n# Enable parallel evaluation\npool = Pool()\ntoolbox.register('map', pool.map)"

        return NotebookCell(cell_type="code", source=format_code(source))

    def _generate_cell_11(self, request: GenerateRequest) -> NotebookCell:
        """Cell 11: Main evolution loop."""
        algo = request.algorithm
        features = request.features

        # Setup
        source = f"""# Initialize population
population = toolbox.population(n={algo.population_size})
"""

        # Statistics
        if features.statistics:
            source += """
# Setup statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
"""

        # Hall of Fame
        if features.hall_of_fame:
            source += f"""
# Hall of Fame
hof = tools.HallOfFame({features.hof_size})
"""

        # Evolution algorithm
        if algo.type == "simple":
            stats_arg = "stats=stats" if features.statistics else "stats=None"
            hof_arg = "halloffame=hof" if features.hall_of_fame else ""
            verbose_arg = f"verbose={str(features.verbose)}"

            source += f"""
# Run evolution
population, logbook = algorithms.eaSimple(
    population, toolbox,
    cxpb={algo.cx_prob}, mutpb={algo.mut_prob},
    ngen={algo.generations},
    {stats_arg}, {hof_arg}, {verbose_arg}
)
"""
        elif algo.type == "mu_plus_lambda":
            mu = algo.mu or algo.population_size
            lambda_ = algo.lambda_ or algo.population_size
            stats_arg = "stats=stats" if features.statistics else "stats=None"
            hof_arg = "halloffame=hof" if features.hall_of_fame else ""

            source += f"""
# Run mu+lambda evolution
population, logbook = algorithms.eaMuPlusLambda(
    population, toolbox,
    mu={mu}, lambda_={lambda_},
    cxpb={algo.cx_prob}, mutpb={algo.mut_prob},
    ngen={algo.generations},
    {stats_arg}, {hof_arg}, verbose={str(features.verbose)}
)
"""
        else:
            # Custom loop
            source += f"""
# Custom evolution loop
for gen in range({algo.generations}):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < {algo.cx_prob}:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation
    for mutant in offspring:
        if random.random() < {algo.mut_prob}:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate individuals with invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring
"""

        return NotebookCell(cell_type="code", source=format_code(source))

    def _generate_cell_12(self, request: GenerateRequest) -> NotebookCell:
        """Cell 12: Results and plotting."""
        features = request.features
        source = ""

        if features.hall_of_fame:
            source += """# Display best individuals
print("\\nBest individuals:")
for i, ind in enumerate(hof, 1):
    print(f"{i}. Fitness: {ind.fitness.values[0]:.6f}")
    print(f"   Solution: {ind[:5]}..." if len(ind) > 5 else f"   Solution: {ind}")
"""

        if features.statistics:
            source += """
# Display final statistics
print("\\nFinal Statistics:")
record = logbook[-1]
print(f"Generation: {record['gen']}")
print(f"Min: {record['min']:.6f}")
print(f"Avg: {record['avg']:.6f}")
print(f"Max: {record['max']:.6f}")
print(f"Std: {record['std']:.6f}")
"""

        if features.plotting:
            source += """
# Plot fitness evolution
gen = logbook.select("gen")
fit_mins = logbook.select("min")
fit_avgs = logbook.select("avg")

plt.figure(figsize=(10, 6))
plt.plot(gen, fit_mins, label="Minimum")
plt.plot(gen, fit_avgs, label="Average")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness Evolution")
plt.legend()
plt.grid(True)
plt.show()
"""

        if features.checkpoint:
            source += """
# Save checkpoint
with open('checkpoint.pkl', 'wb') as f:
    pickle.dump((population, hof, logbook), f)
print("\\nCheckpoint saved to 'checkpoint.pkl'")
"""

        if not source:
            source = "# Evolution complete\nprint('Evolution finished successfully')"

        return NotebookCell(cell_type="code", source=format_code(source))
