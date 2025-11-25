"""Request parser for flexible GenerateRequest format."""

from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


class RequestParser:
    """Parses flexible GenerateRequest into standardized format."""

    # Known fields that are explicitly handled
    KNOWN_FIELDS = {
        'session_id', 'problemName', 'goalDescription', 'fitnessDescription',
        'objectiveFunction', 'objectiveType', 'formalEquation',
        'solutionRepresentation', 'solutionSize', 'domainOfVariables',
        'constraintHandling', 'constraints', 'selectionMethod',
        'crossoverOperator', 'crossoverProbability', 'mutationOperator',
        'mutationProbability', 'customOperators', 'populationSize',
        'numGenerations', 'terminationCondition', 'terminationOther',
        'elitism', 'evaluationBudget', 'knownHeuristics', 'exampleSolutions',
        'outputBestSolution', 'outputProgressLog', 'outputVisualization',
        'executionMode', 'problem', 'algorithm', 'operators', 'features',
        'preferences'
    }

    @staticmethod
    def parse_domain_of_variables(domain_str: Optional[str]) -> tuple[list[float], list[float]]:
        """
        Parse domain string into lower and upper bounds.

        Examples:
            "[0,19]" -> ([0], [19])
            "[-5.12, 5.12]" -> ([-5.12], [5.12])
            "[0,1,0,1]" -> ([0,0], [1,1])
        """
        if not domain_str:
            return ([0], [1])

        try:
            # Try to parse as JSON array
            if domain_str.startswith("[") and domain_str.endswith("]"):
                values = json.loads(domain_str)
                if len(values) == 2:
                    return ([values[0]], [values[1]])
                elif len(values) % 2 == 0:
                    # Assume alternating lower/upper bounds
                    mid = len(values) // 2
                    return (values[:mid], values[mid:])
        except:
            pass

        # Default
        return ([0], [1])

    @staticmethod
    def parse_solution_size(size_str: Optional[str], default: int = 10) -> int:
        """Parse solution size string to integer."""
        if not size_str:
            return default

        try:
            return int(size_str)
        except ValueError:
            logger.warning(f"Could not parse solution size '{size_str}', using default {default}")
            return default

    @staticmethod
    def parse_probability(prob_str: Optional[str], default: float = 0.5) -> float:
        """Parse probability string to float."""
        if not prob_str:
            return default

        try:
            val = float(prob_str)
            return max(0.0, min(1.0, val))  # Clamp to [0,1]
        except ValueError:
            logger.warning(f"Could not parse probability '{prob_str}', using default {default}")
            return default

    @staticmethod
    def parse_int_value(val_str: Optional[str], default: int = 100) -> int:
        """Parse integer value string."""
        if not val_str:
            return default

        try:
            return int(val_str)
        except ValueError:
            logger.warning(f"Could not parse integer '{val_str}', using default {default}")
            return default

    @staticmethod
    def extract_structured_data(request: Any) -> Dict[str, Any]:
        """
        Extract and normalize data from flexible GenerateRequest.

        Returns a structured dictionary suitable for LLM prompting.
        """
        data = {}

        # Problem information
        data["problem_name"] = getattr(request, "problemName", None) or "Optimization Problem"
        data["goal_description"] = getattr(request, "goalDescription", None) or ""
        data["fitness_description"] = getattr(request, "fitnessDescription", None) or ""
        data["objective_function"] = getattr(request, "objectiveFunction", None) or "minimize fitness"
        data["objective_type"] = getattr(request, "objectiveType", None) or "minimization"
        data["formal_equation"] = getattr(request, "formalEquation", None) or ""

        # Solution representation
        data["solution_representation"] = getattr(request, "solutionRepresentation", None) or "real-valued"
        solution_size_str = getattr(request, "solutionSize", None)
        data["solution_size"] = RequestParser.parse_solution_size(solution_size_str, default=10)

        # Domain
        domain_str = getattr(request, "domainOfVariables", None)
        lower, upper = RequestParser.parse_domain_of_variables(domain_str)
        # Expand bounds to match solution size if needed
        if len(lower) == 1:
            lower = lower * data["solution_size"]
        if len(upper) == 1:
            upper = upper * data["solution_size"]
        data["lower_bounds"] = lower
        data["upper_bounds"] = upper

        # Constraints
        data["constraint_handling"] = getattr(request, "constraintHandling", None) or ""
        data["constraints"] = getattr(request, "constraints", None) or ""

        # Operators
        data["selection_method"] = getattr(request, "selectionMethod", None) or "tournament"
        data["crossover_operator"] = getattr(request, "crossoverOperator", None) or "blend"
        data["mutation_operator"] = getattr(request, "mutationOperator", None) or "gaussian"

        cx_prob_str = getattr(request, "crossoverProbability", None)
        data["crossover_probability"] = RequestParser.parse_probability(cx_prob_str, default=0.7)

        mut_prob_str = getattr(request, "mutationProbability", None)
        data["mutation_probability"] = RequestParser.parse_probability(mut_prob_str, default=0.2)

        data["custom_operators"] = getattr(request, "customOperators", None) or ""

        # Algorithm parameters
        pop_size_str = getattr(request, "populationSize", None)
        data["population_size"] = RequestParser.parse_int_value(pop_size_str, default=100)

        num_gen_str = getattr(request, "numGenerations", None)
        data["num_generations"] = RequestParser.parse_int_value(num_gen_str, default=50)

        data["termination_condition"] = getattr(request, "terminationCondition", None) or "maxGenerations"
        data["termination_other"] = getattr(request, "terminationOther", None) or ""

        data["elitism"] = getattr(request, "elitism", None) or ""
        data["evaluation_budget"] = getattr(request, "evaluationBudget", None) or ""
        data["known_heuristics"] = getattr(request, "knownHeuristics", None) or ""
        data["example_solutions"] = getattr(request, "exampleSolutions", None) or ""

        # Output options
        data["output_best_solution"] = getattr(request, "outputBestSolution", True)
        data["output_progress_log"] = getattr(request, "outputProgressLog", False)
        data["output_visualization"] = getattr(request, "outputVisualization", False)

        data["execution_mode"] = getattr(request, "executionMode", None) or "local"

        # Extract extra fields as 'other_specifications'
        other_specifications = {}

        # Get all fields from the request (if it's a Pydantic model or has __dict__)
        if hasattr(request, 'model_dump'):
            # Pydantic v2
            all_fields = request.model_dump()
        elif hasattr(request, 'dict'):
            # Pydantic v1
            all_fields = request.dict()
        elif hasattr(request, '__dict__'):
            # Regular object
            all_fields = request.__dict__
        else:
            all_fields = {}

        # Filter out known fields and None values
        for field_name, field_value in all_fields.items():
            if field_name not in RequestParser.KNOWN_FIELDS and field_value is not None:
                other_specifications[field_name] = field_value
                logger.info(f"Found extra field '{field_name}': {field_value}")

        # Add other_specifications to data if any were found
        if other_specifications:
            data["other_specifications"] = other_specifications
            logger.info(f"Added {len(other_specifications)} extra fields to other_specifications")

        return data
