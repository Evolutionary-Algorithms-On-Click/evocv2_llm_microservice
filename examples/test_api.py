"""Test script for EVOC DEAP Agent API."""

import requests
import json
from typing import Dict, Any, Optional
from datetime import datetime


BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_health():
    """Test health endpoint."""
    print_section("Health Check")

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200, "Health check failed"
    return True


def test_root():
    """Test root endpoint."""
    print_section("Root Endpoint")

    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200, "Root endpoint failed"
    return True


def test_generate_flexible():
    """Test notebook generation with flexible input format."""
    print_section("Generate Notebook (Flexible Format)")

    # Using the new flexible format
    request_data = {
        "session_id": f"test-session-{datetime.now().timestamp()}",
        "problemName": "Traveling Salesman Problem",
        "goalDescription": "Find the shortest route visiting all cities exactly once",
        "fitnessDescription": "Minimize total distance traveled",
        "objectiveFunction": "Sum of distances between consecutive cities in the tour",
        "objectiveType": "minimization",
        "solutionRepresentation": "Permutation of city indices",
        "solutionSize": "20 cities",
        "domainOfVariables": "Permutation of integers 0-19",
        "selectionMethod": "Tournament selection with size 3",
        "crossoverOperator": "Ordered crossover (OX)",
        "crossoverProbability": "0.7",
        "mutationOperator": "Swap mutation",
        "mutationProbability": "0.2",
        "populationSize": "100",
        "numGenerations": "50",
        "terminationCondition": "Fixed number of generations",
        "elitism": "Keep top 10 individuals",
        "outputBestSolution": True,
        "outputProgressLog": True,
        "outputVisualization": True
    }

    response = requests.post(
        f"{BASE_URL}/v1/generate",
        json=request_data,
        timeout=120
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Session ID: {result['session_id']}")
        print(f"Number of cells: {len(result['notebook']['cells'])}")
        print(f"Message: {result['message']}")

        # Print cell names
        print("\nGenerated cells:")
        for i, cell in enumerate(result['notebook']['cells']):
            cell_name = cell.get('cell_name', f'Cell {i}')
            lines = len(cell['source'].split('\n'))
            print(f"  {i}. {cell_name} ({lines} lines)")

        return result['session_id'], result['notebook']
    else:
        print(f"Error: {json.dumps(response.json(), indent=2)}")
        return None, None


def test_generate_simple():
    """Test notebook generation with simple sphere problem."""
    print_section("Generate Notebook (Simple Sphere)")

    request_data = {
        "session_id": f"sphere-{datetime.now().timestamp()}",
        "problemName": "Sphere Function Optimization",
        "goalDescription": "Minimize the sphere function",
        "objectiveFunction": "sum(x^2 for x in individual)",
        "objectiveType": "minimization",
        "solutionRepresentation": "List of real numbers",
        "solutionSize": "10 dimensions",
        "domainOfVariables": "[-5.0, 5.0] for each dimension",
        "selectionMethod": "Tournament selection",
        "crossoverOperator": "Blend crossover",
        "mutationOperator": "Gaussian mutation",
        "populationSize": "50",
        "numGenerations": "30"
    }

    response = requests.post(
        f"{BASE_URL}/v1/generate",
        json=request_data,
        timeout=120
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Session ID: {result['session_id']}")
        print(f"Cells generated: {len(result['notebook']['cells'])}")

        # Show first cell (imports)
        print("\nFirst cell (Imports):")
        print(result['notebook']['cells'][0]['source'][:300] + "...")

        return result['session_id'], result['notebook']
    else:
        print(f"Error: {json.dumps(response.json(), indent=2)}")
        return None, None


def test_modify(session_id: str, notebook: Dict[str, Any]):
    """Test notebook modification."""
    print_section("Modify Notebook")

    request_data = {
        "instruction": "Change the population size to 200 and generations to 100. Use polynomial bounded mutation instead.",
        "current_notebook": notebook
    }

    response = requests.post(
        f"{BASE_URL}/v1/sessions/{session_id}/modify",
        json=request_data,
        timeout=120
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Changes made:")
        for change in result['changes_made']:
            print(f"  - {change}")
        print(f"Message: {result['message']}")
        return result['notebook']
    else:
        print(f"Error: {json.dumps(response.json(), indent=2)}")
        return None


def test_fix(session_id: str, notebook: Dict[str, Any]):
    """Test notebook fixing (simulated error)."""
    print_section("Fix Notebook")

    # Simulate a traceback
    traceback = """
Traceback (most recent call last):
  File "<cell>", line 10, in <module>
    toolbox.register('evaluate', evaluate)
NameError: name 'evaluate' is not defined
"""

    request_data = {
        "traceback": traceback,
        "current_notebook": notebook,
        "context": "Function registration happening before function definition"
    }

    response = requests.post(
        f"{BASE_URL}/v1/sessions/{session_id}/fix",
        json=request_data,
        timeout=120
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Fixes applied:")
        for fix in result['fixes_applied']:
            print(f"  - {fix}")
        print(f"Validation passed: {result['validation_passed']}")
        print(f"Message: {result['message']}")
        return result['notebook']
    else:
        print(f"Error: {json.dumps(response.json(), indent=2)}")
        return None


def test_get_session(session_id: str):
    """Test get session endpoint."""
    print_section("Get Session")

    response = requests.get(f"{BASE_URL}/v1/sessions/{session_id}")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Session ID: {result['session_id']}")
        print(f"Created: {result['created_at']}")
        print(f"Updated: {result['updated_at']}")
        print(f"History entries: {len(result['history'])}")

        if result['history']:
            print("\nHistory:")
            for i, entry in enumerate(result['history'][-3:], 1):
                print(f"  {i}. {entry.get('operation', 'unknown')} at {entry.get('timestamp', 'unknown')}")
    else:
        print(f"Session not found or error: {response.status_code}")


def test_list_sessions():
    """Test list sessions endpoint."""
    print_section("List Sessions")

    response = requests.get(f"{BASE_URL}/v1/sessions")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Total sessions: {len(result['sessions'])}")

        for session in result['sessions'][:5]:
            print(f"  - {session['session_id']} (created: {session['created_at']})")
    else:
        print(f"Error: {response.status_code}")


def test_delete_session(session_id: str):
    """Test delete session endpoint."""
    print_section("Delete Session")

    response = requests.delete(f"{BASE_URL}/v1/sessions/{session_id}")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        print(f"Session {session_id} deleted successfully")
    else:
        print(f"Error: {response.status_code}")


def run_full_workflow():
    """Run a complete workflow test."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "EVOC DEAP Agent - Full Test" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")

    try:
        # Test 1: Health and root
        test_health()
        test_root()

        # Test 2: Generate with flexible format
        session_id, notebook = test_generate_flexible()

        if session_id and notebook:
            # Test 3: Modify notebook
            modified_notebook = test_modify(session_id, notebook)

            # Test 4: Get session details
            test_get_session(session_id)

            # Test 5: List all sessions
            test_list_sessions()

        # Test 6: Generate simple problem
        simple_session_id, simple_notebook = test_generate_simple()

        if simple_session_id and simple_notebook:
            # Test 7: Get the simple session
            test_get_session(simple_session_id)

        print_section("✓ All Tests Passed")

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the service.")
        print("   Make sure the service is running on http://localhost:8000")
        print("   Run: docker-compose up -d")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    run_full_workflow()


if __name__ == "__main__":
    main()
