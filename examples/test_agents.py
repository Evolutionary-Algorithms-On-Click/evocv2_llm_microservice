"""Test script for the new modular system with LLM-based generation."""

import json
from app.models import GenerateRequest
from app.agents.generator_new import NotebookGenerator
from app.utils.notebook_exporter import export_for_testing, mock_test_execution


def test_flexible_json_request():
    """Test the new flexible JSON request format."""

    # Example request matching the user's format
    request_data = {
        "session_id": "test_session_123",
        "constraintHandling": "",
        "constraints": "",
        "crossoverOperator": "blend",
        "crossoverProbability": "0.7",
        "customOperators": "",
        "domainOfVariables": "[0,19]",
        "elitism": "",
        "evaluationBudget": "",
        "exampleSolutions": "",
        "executionMode": "local",
        "fitnessDescription": "minimize the sum of distances in the traveling salesman problem",
        "formalEquation": "",
        "goalDescription": "Find the shortest route visiting all cities",
        "knownHeuristics": "",
        "mutationOperator": "gaussian",
        "mutationProbability": "0.2",
        "numGenerations": "50",
        "objectiveFunction": "f(x) = sum(distances between cities)",
        "objectiveType": "minimization",
        "outputBestSolution": True,
        "outputProgressLog": False,
        "outputVisualization": True,
        "populationSize": "100",
        "problemName": "Traveling Salesman Problem",
        "selectionMethod": "tournament",
        "solutionRepresentation": "permutation",
        "solutionSize": "20",
        "terminationCondition": "maxGenerations",
        "terminationOther": ""
    }

    print("=" * 80)
    print("Testing New System with Flexible JSON Request")
    print("=" * 80)

    # Create request object
    request = GenerateRequest(**request_data)
    print(f"\n✓ Request parsed successfully")
    print(f"  Session ID: {request.session_id}")
    print(f"  Problem: {request.problemName}")
    print(f"  Solution Size: {request.solutionSize}")

    # Generate notebook using new LLM-based generator
    print("\n" + "=" * 80)
    print("Generating Notebook with LLM...")
    print("=" * 80)

    generator = NotebookGenerator()
    notebook = generator.generate(request)

    print(f"\n✓ Notebook generated successfully")
    print(f"  Total cells: {len(notebook.cells)}")

    # Display cell names
    print("\n  Cell structure:")
    for i, cell in enumerate(notebook.cells):
        cell_name = cell.cell_name or f"cell_{i}"
        lines = len(cell.source.split('\n'))
        print(f"    {i}. {cell_name:25} ({lines:3} lines)")

    # Export for testing
    print("\n" + "=" * 80)
    print("Exporting to Python Script...")
    print("=" * 80)

    export_results = export_for_testing(
        notebook=notebook,
        base_filename="test_tsp_problem",
        output_dir="./test_outputs"
    )

    print(f"\n✓ Exported to multiple formats:")
    for format_name, path in export_results.items():
        if format_name.endswith("_content"):
            continue
        print(f"  - {format_name}: {path}")

    # Mock test execution
    print("\n" + "=" * 80)
    print("Running Mock Tests...")
    print("=" * 80)

    test_results = mock_test_execution(export_results["script_content"])

    print(f"\n✓ Test Results:")
    print(f"  Syntax Valid: {test_results['syntax_valid']}")
    print(f"  Imports OK: {test_results['imports_ok']}")
    print(f"  Has Evolution Loop: {test_results['has_evolution_loop']}")
    print(f"  Total Lines: {test_results['line_count']}")

    if test_results['errors']:
        print(f"\n✗ Errors found:")
        for error in test_results['errors']:
            print(f"  - {error}")
    else:
        print(f"\n✓ No syntax errors detected!")

    # Display sample of generated code
    print("\n" + "=" * 80)
    print("Sample of Generated Code (First Cell - Imports):")
    print("=" * 80)
    print(notebook.cells[0].source)

    print("\n" + "=" * 80)
    print("Sample of Generated Code (Evaluate Function):")
    print("=" * 80)
    print(notebook.cells[3].source)

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

    return notebook, export_results


def test_modify_with_cell_type():
    """Test modification with specific cell_type targeting."""

    from app.models import ModifyRequest, NotebookStructure, NotebookCell

    print("\n" + "=" * 80)
    print("Testing Modification with cell_type Targeting")
    print("=" * 80)

    # Create a simple notebook for testing
    cells = [
        NotebookCell(cell_type="code", cell_name="imports", source="import numpy as np"),
        NotebookCell(cell_type="code", cell_name="mutation", source="def mutate(x): return x"),
    ] + [NotebookCell(cell_type="code", cell_name=f"cell_{i}", source="pass") for i in range(10)]

    notebook = NotebookStructure(cells=cells)

    # Create modify request targeting specific cell_type
    modify_request = ModifyRequest(
        instruction="Change the mutation operator to use polynomial bounded mutation",
        current_notebook=notebook,
        cell_type="mutation"  # This is the new field
    )

    print(f"\n✓ Modify request created")
    print(f"  Instruction: {modify_request.instruction}")
    print(f"  Target cell_type: {modify_request.cell_type}")
    print(f"  Current notebook has {len(modify_request.current_notebook.cells)} cells")

    return modify_request


if __name__ == "__main__":
    # Test 1: Flexible JSON request with LLM generation
    notebook, exports = test_flexible_json_request()

    # Test 2: Modification with cell_type
    modify_req = test_modify_with_cell_type()

    print("\n" + "=" * 80)
    print("All Tests Completed Successfully!")
    print("=" * 80)
    print(f"\nGenerated files can be found in: ./test_outputs/")
    print(f"  - Python script: {exports['python_script']}")
    print(f"  - Jupyter notebook: {exports['jupyter_notebook']}")
    print("\nYou can now run the generated Python script to test the evolutionary algorithm!")
