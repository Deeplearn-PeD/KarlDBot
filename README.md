# Karl D. Bot

Karl is a multi-agent reinforcement learning system that produces high-quality data science code.

## Architecture

```
karldbot/
├── agents/           # Agent implementations
│   ├── base.py       # Abstract base agent class
│   ├── koder.py      # Code generation agent
│   ├── reviewer.py   # Code review agent
│   ├── analyst.py    # Data analysis agent
│   ├── visualizer.py # Visualization agent
│   └── tester.py     # Testing agent
├── environment/      # RL environment
│   ├── core.py       # Environment class
│   └── problem.py    # Problem definition
├── llm/              # LLM interfaces
│   ├── interface.py  # Sync/Async LLM wrapper
│   └── prompts.py    # Prompt management
├── models/           # Pydantic data models
│   ├── config.py     # Configuration models
│   ├── state.py      # State and workflow models
│   ├── messages.py   # Agent communication
│   └── actions.py    # Action enumerations
├── orchestration/    # Agent coordination
│   └── coordinator.py
├── report/           # Report generation
│   └── generator.py
└── cli.py            # Command-line interface
```

## Installation

```bash
pip install karldbot
```

Or with uv:

```bash
uv pip install -e .
```

## Configuration

Create a `problem.yaml` file:

```yaml
problem_name: Climate Analysis
data_source: data.csv
description: Analyze climate data to identify trends and patterns.
llm_model: gpt-4o
max_iterations: 50
```

## Usage

### Train the agent

```bash
karl train --config problem.yaml
```

### View the report

```bash
karl view_report --config problem.yaml
```

## Multi-Agent System

KarlDBot uses multiple specialized agents:

| Agent | Role |
|-------|------|
| **Koder** | Writes, debugs, and optimizes code |
| **CodeReviewer** | Reviews code quality and approves solutions |
| **DataAnalyst** | Performs exploratory data analysis |
| **Visualizer** | Creates visualizations and charts |
| **Tester** | Generates and runs tests |

## Workflow States

```
INIT → CODING → REVIEWING → DEBUGGING → CODING
                   ↓
              OPTIMIZING → CODING
                   ↓
              COMPLETED
```

## Development

### Run tests

```bash
pytest tests/
```

### Type checking

```bash
mypy karldbot/
```

### Linting

```bash
ruff check karldbot/
```

## License

GPLv3
