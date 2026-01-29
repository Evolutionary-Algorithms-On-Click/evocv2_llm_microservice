from typing import Dict, Any, Optional

def get_complete_notebook_prompt(problem_data: Dict[str, Any]) -> str:
    # Extract other_specifications if present
    other_specs = problem_data.get('other_specifications', {})
    other_specs_text = ""
    if other_specs:
        other_specs_text = "\n**Additional Specifications:**\n"
        for key, value in other_specs.items():
            other_specs_text += f"- {key}: {value}\n"



    # TODO: Change prompt according to relative path in Docker file system (in NOTE: )
    prompt = f"""Generate a complete 12-cell DEAP evolutionary algorithm notebook.

NOTE: Load dataset for input for the algorithm from mnt/user_data/session-id/{problem_data.get('data_source', 'N/A')}  with headers {problem_data.get('data_headers', 'N/A')} if applicable. Solve the problem using the provided dataset header.    
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
- Size: {problem_data.get('solution_size', 10)} variables mention as DIMENSIONS
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
NOTE: You must use the following variable in Cell 1 (config) with the exact same variable name case sensitive
USER_ID, NOTEBOOK_ID, POP_SIZE (population size), CX_PROB (crossover probability), MUT_PROB (mutation probability)

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
6. Generate a list of requirements (pip packages) needed to run this code

Generate the complete notebook as a JSON object with 'cells' array and 'requirements' string."""
    return prompt

def get_single_cell_prompt(
    cell_index: int,
    cell_name: str,
    cell_description: str,
    problem_data: Dict[str, Any],
    context: Optional[Dict[str, str]] = None
) -> str:
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
    prompt += _get_cell_specific_instructions(cell_index, problem_data, context)

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
    cell_index: int,
    problem_data: Dict[str, Any],
    context: Optional[Dict[str, str]] = None
) -> str:
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
{' - Create matplotlib plot showing fitness evolution over generations' if show_viz else ''}
- Use plt.figure(), plt.plot(), plt.legend(), plt.show()
"""

    return ""

def get_system_prompt_complete_generation() -> str:
    return """You are an expert in DEAP (Distributed Evolutionary Algorithms in Python).
Generate a complete, consistent 12-cell DEAP notebook.

CRITICAL REQUIREMENTS:
1. Generate ALL imports ONLY in the 'imports' cell (cell 0)
2. Do NOT repeat imports in other cells
3. Use creator.create() ONLY ONCE in the 'creator' cell (cell 2) - never redefine fitness or individual classes
4. Ensure all cells work together as a cohesive notebook
5. Reference variables and functions defined in previous cells
6. Generate clean, efficient, executable Python code"""

def get_system_prompt_cell_generation() -> str:
    return "You are an expert in DEAP (Distributed Evolutionary Algorithms in Python). Generate clean, efficient, and correct DEAP code for evolutionary algorithms."