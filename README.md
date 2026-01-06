# Bio-inspired Optimization Benchmark Runner

A professional benchmark runner for bio-inspired optimization algorithms. Run reproducible experiments on benchmark functions, TSP, and scheduling problems with standardized interfaces, automatic artifact saving, and aggregated results.

## Algorithms

- **GA (Genetic Algorithm)**: Binary encoding for continuous optimization, permutation encoding for TSP
- **PSO (Particle Swarm Optimization)**: Swarm-based continuous optimization
- **ACO (Ant Colony Optimization)**: Pheromone-based combinatorial optimization for TSP
- **GWO (Grey Wolf Optimizer)**: Wolf pack hierarchy for scheduling and continuous problems

## Tasks

| Task | Variants | Description |
|------|----------|-------------|
| `functions` | beale, easom, rastrigin | Standard benchmark functions for continuous optimization |
| `tsp` | random, grid | Travelling Salesman Problem with different graph types |
| `scheduling` | default | Class scheduling with professor availability and room constraints |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bioopt-bench.git
cd bioopt-bench

# Install in development mode
pip install -e ".[dev]"

# Or install with notebook support
pip install -e ".[dev,notebook]"
```

## Quickstart

### List available algorithms and tasks

```bash
python -m bioopt_bench list
```

### Run a single benchmark

```bash
# Function optimization with PSO
python -m bioopt_bench run --task functions --fn beale --algo pso --iters 200 --seed 42 --save

# TSP with ACO
python -m bioopt_bench run --task tsp --graph random --n-cities 50 --algo aco --iters 100 --seed 42 --save

# Class scheduling with GWO
python -m bioopt_bench run --task scheduling --algo gwo --iters 300 --seed 42 --save
```

### Run a benchmark suite

```bash
# Run default suite with 3 repetitions
python -m bioopt_bench suite --suite default --repeat 3 --save

# Run quick smoke test
python -m bioopt_bench suite --suite quick --repeat 1 --save
```

## CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `list` | List available algorithms, tasks, and suites |
| `run` | Run a single benchmark |
| `suite` | Run a predefined benchmark suite |

### Run Options

```
--task {functions,tsp,scheduling}   Task type (required)
--variant / --fn / --graph          Task variant (required)
--algo {ga,pso,aco,gwo}             Algorithm (required)
--iters INT                         Number of iterations (default: 100)
--seed INT                          Random seed (default: 42)
--save                              Save results and artifacts
--reports-dir PATH                  Output directory (default: reports/)
```

### Task-specific options

```
# Functions
--dim INT                           Dimension for Rastrigin (default: 10)

# TSP
--n-cities INT                      Number of cities (default: 50)

# Scheduling
--generate-demo                     Use demo scheduling data
```

### Algorithm-specific options

```
--population-size INT               GA/GWO population size
--mutation-rate FLOAT               GA mutation rate
--n-particles INT                   PSO particle count
--n-ants INT                        ACO ant count
--n-wolves INT                      GWO wolf count
```

## Output Structure

Each benchmark run with `--save` creates:

```
reports/
├── results.csv                      # Aggregated results (appended per run)
└── runs/
    └── 20240115_143022_a1b2c3/     # Run ID: YYYYMMDD_HHMMSS_hash
        ├── config.json              # Full configuration
        ├── metrics.json             # Final metrics
        ├── curves.csv               # Convergence history
        ├── artifacts/
        │   └── best_solution.json   # Best solution found
        └── figures/
            └── fitness_convergence.png
```

### results.csv columns

| Column | Description |
|--------|-------------|
| timestamp | ISO timestamp |
| run_id | Unique run identifier |
| task | Task type |
| task_variant | Task variant |
| algorithm | Algorithm name |
| seed | Random seed used |
| iterations | Number of iterations |
| best_fitness | Best fitness achieved |
| final_fitness | Final fitness value |
| runtime_s | Runtime in seconds |
| extra_metrics_json | Algorithm-specific metrics |

## Predefined Suites

| Suite | Description |
|-------|-------------|
| `default` | Full benchmark: all tasks with primary algorithms |
| `functions-only` | All function benchmarks |
| `tsp-only` | All TSP benchmarks |
| `scheduling-only` | Scheduling benchmarks |
| `quick` | Fast smoke test (~1 minute) |

## Example Runs

### 1. Beale function with PSO

```bash
python -m bioopt_bench run --task functions --fn beale --algo pso --iters 200 --seed 0 --save
```

Output:
```
Running PSO on beale...
  Task: functions
  Iterations: 200
  Seed: 0

Results:
  Run ID: 20240115_143022_a1b2c3
  Best Fitness: 0.000003
  Final Fitness: 0.000003
  Runtime: 0.245s
```

### 2. TSP with 50 cities using ACO

```bash
python -m bioopt_bench run --task tsp --graph random --n-cities 50 --algo aco --iters 100 --seed 1 --save
```

### 3. Full suite with multiple seeds

```bash
python -m bioopt_bench suite --suite default --repeat 5 --save
```

## Extending

### Add a new algorithm

1. Create `bioopt_bench/algorithms/my_algo.py`:

```python
class MyAlgorithm:
    def __init__(self, param1=default_value):
        self.param1 = param1

    def run(self, task, seed, iterations=100, **kwargs):
        # ... implementation ...
        return solution, history, meta
```

2. Register in `bioopt_bench/registry.py`:

```python
from .algorithms.my_algo import MyAlgorithm

ALGORITHMS["my_algo"] = MyAlgorithm
ALGORITHM_TASK_COMPATIBILITY["my_algo"] = ["functions", "tsp"]
```

### Add a new task

1. Create `bioopt_bench/tasks/my_task.py`:

```python
class MyTask:
    task_type = "my_task"
    name = "my_task"

    def evaluate(self, solution):
        # ... return fitness ...
        pass

    def get_config(self):
        return {"name": self.name, ...}
```

2. Register in `bioopt_bench/registry.py`:

```python
TASKS["my_task"] = {"variant1": MyTask}
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -q

# Format code
black .

# Lint
ruff check .
```

## Project Structure

```
.
├── bioopt_bench/
│   ├── __init__.py
│   ├── cli.py              # CLI entrypoint
│   ├── registry.py         # Algorithm/task registry
│   ├── reporting.py        # Artifact saving
│   ├── benchmark.py        # Run orchestration
│   ├── common/
│   │   ├── seeding.py      # Reproducibility
│   │   └── types.py        # Data classes
│   ├── algorithms/
│   │   ├── ga.py
│   │   ├── pso.py
│   │   ├── aco.py
│   │   └── gwo.py
│   └── tasks/
│       ├── functions.py    # Beale, Easom, Rastrigin
│       ├── tsp.py          # TSP generators
│       └── scheduling.py   # Class scheduling
├── notebooks/
│   └── demo.ipynb          # Interactive demo
├── tests/
├── reports/                # Generated outputs (gitignored)
├── pyproject.toml
└── README.md
```

## License

MIT License - see [LICENSE](LICENSE) for details.
