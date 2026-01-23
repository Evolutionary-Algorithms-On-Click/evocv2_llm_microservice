# EVOC DEAP Agent

**Python service for generating and maintaining purely functional DEAP evolutionary algorithm code in currently rigid 12-cell Jupyter notebook format.**

# TODO integrate with VolPE system

## Overview

EVOC DEAP Agent is an intelligent service that creates, modifies, and fixes DEAP code following a strict functional programming style with exactly 12 cells, perfect for Jupyter notebooks.

### Key Features

- **Flexible Input**: Accepts any JSON structure - extra fields automatically sent to LLM as "other_specifications"
- **Generate**: Create complete DEAP notebooks from structured JSON specifications
- **Modify**: Update existing notebooks via natural language instructions
- **Fix**: Automatically repair broken code using error tracebacks
- **Smart Dependencies**: Hybrid static + LLM-based dependency detection
- **Structured Output**: 100% JSON via Instructor + Pydantic (no markdown)
- **Memory Layer**: Mem0 integration for session history and preferences
- **LangGraph Workflow**: State management with validation loops


## Architecture

### Tech Stack

- **FastAPI**: For REST API
- **LangGraph**: Workflow orchestration with state management
- **Instructor + Pydantic**: Guaranteed structured LLM outputs
- **Mem0**: Session memory and preference tracking
- **Groq**: Fast LLM inference (compatible with ChatGroq)
- **Python 3.11+**

### 12-Cell Structure

Every generated notebook follows this exact structure:

1. **Cell 0**: Imports (DEAP, NumPy, Matplotlib, etc.)
2. **Cell 1**: Problem configuration (dimensions, bounds)
3. **Cell 2**: Creator setup (`creator.create` for Fitness and Individual)
4. **Cell 3**: Evaluate function (`def evaluate`)
5. **Cell 4**: Mate/crossover function (`def mate`)
6. **Cell 5**: Mutation function (`def mutate`)
7. **Cell 6**: Selection function (`def select`)
8. **Cell 7**: Additional operators (optional)
9. **Cell 8**: Initialization functions (`def create_individual`)
10. **Cell 9**: Toolbox registration (`toolbox.register()` calls)
11. **Cell 10**: Main evolution loop (eaSimple, eaMuPlusLambda, or custom)
12. **Cell 11**: Results, plotting, statistics

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Groq API key (get one at [groq.com](https://groq.com))

### Build and Run

1. **Clone and navigate to the project:**
   ```bash
   cd evoc-deap-agent
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

3. **Build the Docker image:**
   ```bash
   docker build -t evoc-deap-agent .
   ```

4. **Run the container:**
   ```bash
   docker run -p 8000:8000 --env-file .env evoc-deap-agent
   ```

   **Or use Docker Compose:**
   ```bash
   docker-compose up
   ```

5. **Verify it's running:**
   ```bash
   curl http://localhost:8000/health
   ```



## Development

### Running Locally Without Docker

1. **Install Python 3.11+ (if not already installed)**
   ```bash
   python --version  # Should be 3.11 or higher
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv

   # On Windows:
   venv\Scripts\activate

   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and add your API keys:


5. **Run the service:**
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   Or directly:
   ```bash
   python -m app.main
   ```

6. **Access the API:**
   - **API Root:** http://localhost:8000
   - **Swagger UI:** http://localhost:8000/docs
   - **ReDoc:** http://localhost:8000/redoc
   - **Health Check:** http://localhost:8000/health

7. **Test the API:**
   ```bash
   curl http://localhost:8000/health
   ```

   Expected response:
   ```json
   {"status": "healthy", "version": "2.0.0"}
   ```




