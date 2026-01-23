import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms


# config
DIMENSIONS = 10
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0
POP_SIZE = 100
NGEN = 1000
CXPB = 0.8  # crossover probability
MUTPB = 0.1  # mutation probability
random.seed(42)
np.random.seed(42)

# creator
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


# evaluate
def evaluate(individual):
    """Mock energy function for a protein conformation.
    The real function would compute bonded, angle, dihedral and non-bonded terms.
    Here we use a simple quadratic term plus a small interaction term.
    """
    arr = np.array(individual)
    energy = np.sum(arr**2) + 0.5 * np.sum(np.abs(np.diff(arr)))
    return (energy,)


# crossover
def mate(ind1, ind2):
    """Blend crossover (cxBlend) with alpha=0.5."""
    tools.cxBlend(ind1, ind2, alpha=0.5)
    return ind1, ind2


# mutation
def mutate(individual):
    """Gaussian mutation with clipping to the search bounds."""
    tools.mutGaussian(individual, mu=0.0, sigma=0.1, indpb=0.2)
    # Clip to bounds
    for i in range(len(individual)):
        if individual[i] < LOWER_BOUND:
            individual[i] = LOWER_BOUND
        elif individual[i] > UPPER_BOUND:
            individual[i] = UPPER_BOUND
    return (individual,)


# selection
def select(population, k):
    """Tournament selection with tournsize=3."""
    return tools.selTournament(population, k, tournsize=3)


# additional_operators
# Placeholder for any future custom operators (e.g., elitism, crowding).
# Currently the standard DEAP operators are sufficient for this demo.


# initialization
def create_individual():
    """Create a random individual within the defined bounds."""
    return [random.uniform(LOWER_BOUND, UPPER_BOUND) for _ in range(DIMENSIONS)]


# toolbox_registration
toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_float", random.uniform, LOWER_BOUND, UPPER_BOUND)
# Structure initializers
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=DIMENSIONS
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# Operator registrations
toolbox.register("evaluate", evaluate)
toolbox.register("mate", mate)
toolbox.register("mutate", mutate)
toolbox.register("select", select)

# evolution_loop
pop = toolbox.population(n=POP_SIZE)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)
logbook = tools.Logbook()
logbook.header = ["gen", "nevals"] + stats.fields

for gen in range(NGEN):
    # Selection
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    # Crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Mutation
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate invalid individuals
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Replace population
    pop[:] = offspring

    # Record statistics
    record = stats.compile(pop)
    logbook.record(gen=gen, nevals=len(invalid_ind), **record)
    if gen % 100 == 0:
        print(logbook.stream)

# results_and_plots
best_ind = tools.selBest(pop, 1)[0]
print("Best individual:", best_ind)
print("Best fitness (minimum energy):", best_ind.fitness.values[0])

# Plot convergence if matplotlib is available
if True:
    generations = logbook.select("gen")
    min_fitness = logbook.select("min")
    plt.figure(figsize=(8, 5))
    plt.plot(generations, min_fitness, label="Minimum Energy")
    plt.xlabel("Generation")
    plt.ylabel("Energy")
    plt.title("Evolution of Minimum Energy over Generations")
    plt.legend()
    plt.grid(True)
    plt.show()
