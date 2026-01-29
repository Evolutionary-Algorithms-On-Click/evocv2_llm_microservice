"""Pydantic models for structured input/output."""

from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field


class ProblemBounds(BaseModel):
    """Variable bounds for the optimization problem."""
    lower: List[float] = Field(..., description="Lower bounds for each dimension")
    upper: List[float] = Field(..., description="Upper bounds for each dimension")


class ObjectiveFunction(BaseModel):
    """Objective function specification."""
    type: Literal["builtin", "custom"] = Field(default="builtin")
    name: Optional[str] = Field(None, description="Name of builtin function (sphere, rastrigin, etc.)")
    code: Optional[str] = Field(None, description="Custom Python code for objective function")
    minimize: bool = Field(default=True, description="Whether to minimize (True) or maximize (False)")


class ProblemConfig(BaseModel):
    """Problem specification."""
    dimensions: str = Field(..., description="Number of dimensions")
    bounds: ProblemBounds
    objective: ObjectiveFunction


class OperatorConfig(BaseModel):
    """Configuration for genetic operators."""
    selection: str = Field(default="selTournament", description="Selection operator")
    selection_params: Dict[str, Any] = Field(default_factory=lambda: {"tournsize": 3})
    crossover: str = Field(default="cxBlend", description="Crossover operator")
    crossover_params: Dict[str, Any] = Field(default_factory=lambda: {"alpha": 0.5})
    mutation: str = Field(default="mutGaussian", description="Mutation operator")
    mutation_params: Dict[str, Any] = Field(default_factory=lambda: {"mu": 0, "sigma": 1, "indpb": 0.2})


class AlgorithmConfig(BaseModel):
    """Evolutionary algorithm configuration."""
    type: Literal["simple", "mu_plus_lambda", "mu_comma_lambda", "custom"] = Field(default="simple")
    population_size: int = Field(default=100, ge=10)
    generations: int = Field(default=50, ge=1)
    cx_prob: float = Field(default=0.7, ge=0, le=1, description="Crossover probability")
    mut_prob: float = Field(default=0.2, ge=0, le=1, description="Mutation probability")
    mu: Optional[int] = Field(None, description="Mu for mu+lambda or mu,lambda")
    lambda_: Optional[int] = Field(None, description="Lambda for mu+lambda or mu,lambda")


class Features(BaseModel):
    """Optional features to include."""
    hall_of_fame: bool = Field(default=True, description="Track best individuals")
    hof_size: int = Field(default=10, ge=1)
    statistics: bool = Field(default=True, description="Collect statistics")
    plotting: bool = Field(default=True, description="Generate plots")
    checkpoint: bool = Field(default=False, description="Save checkpoints")
    parallel: bool = Field(default=False, description="Parallel evaluation")
    verbose: bool = Field(default=True, description="Print progress")


class NotebookCell(BaseModel):
    """Single notebook cell."""
    cell_type: Literal["code", "markdown"] = "code"
    cell_name: Optional[str] = Field(None, description="Descriptive name for the cell (e.g., 'imports', 'mutation', 'plots')")
    source: str
    execution_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NotebookStructure(BaseModel):
    """Complete 12-cell notebook structure."""
    cells: List[NotebookCell] = Field(..., min_length=12, max_length=12)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    requirements: str = Field(default="", description="Newline-separated requirements for the code")


class GenerateRequest(BaseModel):
    """Request to generate a new DEAP notebook - flexible structure."""
    model_config = {"extra": "allow"}  # Allow any additional fields

    # Required identifiers
    user_id: str = Field(..., description="Unique user identifier (Mem0 user_id)")
    notebook_id: str = Field(..., description="Unique notebook identifier (Mem0 run_id)")

    # Flexible fields that might come in various formats
    problemName: Optional[str] = None
    goalDescription: Optional[str] = None
    fitnessDescription: Optional[str] = None
    objectiveFunction: Optional[str] = None
    objectiveType: Optional[str] = None
    formalEquation: Optional[str] = None

    # Dataset details
    dataSource: Optional[str] = None
    dataHeaders: Optional[str] = None



    solutionRepresentation: Optional[str] = None
    solutionSize: Optional[str] = None
    domainOfVariables: Optional[str] = None

    constraintHandling: Optional[str] = None
    constraints: Optional[str] = None

    selectionMethod: Optional[str] = None
    crossoverOperator: Optional[str] = None
    crossoverProbability: Optional[str] = None
    mutationOperator: Optional[str] = None
    mutationProbability: Optional[str] = None
    customOperators: Optional[str] = None

    populationSize: Optional[str] = None
    numGenerations: Optional[str] = None
    terminationCondition: Optional[str] = None
    terminationOther: Optional[str] = None

    elitism: Optional[str] = None
    evaluationBudget: Optional[str] = None
    knownHeuristics: Optional[str] = None
    exampleSolutions: Optional[str] = None

    outputBestSolution: Optional[bool] = None
    outputProgressLog: Optional[bool] = None
    outputVisualization: Optional[bool] = None

    executionMode: Optional[str] = None

    # Legacy structured fields (for backward compatibility)
    problem: Optional[ProblemConfig] = None
    algorithm: Optional[AlgorithmConfig] = None
    operators: Optional[OperatorConfig] = None
    features: Optional[Features] = None
    preferences: Dict[str, Any] = Field(default_factory=dict, description="Additional user preferences")


class ModifyRequest(BaseModel):
    """Request to modify an existing notebook."""
    user_id: str = Field(..., description="Unique user identifier (Mem0 user_id)")
    notebook_id: str = Field(..., description="Notebook identifier (Mem0 run_id)")

    instruction: str = Field(..., description="Natural language instruction for modification")
    notebook: NotebookStructure = Field(..., description="Current notebook to modify")
    cell_name: Optional[str] = Field(None, description="Specific cell to modify (e.g., 'mutate', 'crossover'). If None, notebook-level change.")
    preferences: Dict[str, Any] = Field(default_factory=dict)


class FixRequest(BaseModel):
    """Request to fix a broken notebook."""
    user_id: str = Field(..., description="Unique user identifier (Mem0 user_id)")
    notebook_id: str = Field(..., description="Notebook identifier (Mem0 run_id)")

    traceback: str = Field(..., description="Error traceback from execution")
    notebook: NotebookStructure = Field(..., description="Current notebook with errors")
    context: Optional[str] = Field(None, description="Additional context about the error")


class GenerateResponse(BaseModel):
    """Response from generate endpoint."""
    notebook_id: str = Field(..., description="Notebook ID (same as request)")
    notebook: NotebookStructure
    requirements: str = Field(description="Newline-separated requirements for the code")
    message: str = "Notebook generated successfully"


class ModifyResponse(BaseModel):
    """Response from modify endpoint."""
    notebook_id: str
    notebook: NotebookStructure
    changes_made: List[str] = Field(description="List of changes applied")
    cells_modified: List[int] = Field(description="Indices of modified cells")
    requirements: str = Field(description="Newline-separated requirements for the code")
    message: str = "Notebook modified successfully"


class FixResponse(BaseModel):
    """Response from fix endpoint."""
    notebook_id: str
    notebook: NotebookStructure
    fixes_applied: List[str] = Field(description="List of fixes applied")
    validation_passed: bool
    requirements: str = Field(description="Newline-separated requirements for the code")
    message: str = "Notebook fixed successfully"


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    notebook_id: Optional[str] = None
