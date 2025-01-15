# Bio-inspired Optimization Algorithms

This repository contains implementations of bio-inspired algorithms to solve optimization problems, including benchmark functions and real-world applications. The algorithms implemented are:

## Features
1. **Genetic Algorithm (GA):**
   - Optimization of benchmark functions (Beale and Easom).
   - Solves the Travelling Salesman Problem (TSP) on random and grid graphs.
   - Includes parameter tuning and result comparisons.

2. **Particle Swarm Optimization (PSO):**
   - Applied to benchmark functions (Beale, Easom, and Rastrigin).
   - Visualizes fitness convergence over iterations.

3. **Ant Colony Optimization (ACO):**
   - Used to solve TSP on different graph configurations.
   - Evaluates performance and runtime efficiency.

4. **Grey Wolf Optimizer (GWO):**
   - Adapted for a class scheduling problem.
   - Optimizes professor availability, classroom capacity, and schedule conflicts.
   - Includes detailed fitness evolution and conflict analysis.

## Files
- `Deber02.ipynb`: Jupyter Notebook with all implementations and results.
- Example input datasets for TSP and scheduling problems.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bio-inspired-optimization.git
   cd bio-inspired-optimization
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook to execute specific algorithms:
   ```bash
   jupyter notebook Deber02.ipynb
   ```

## Results
- Fitness evolution for each algorithm.
- Detailed comparison of GA vs. ACO on TSP graphs.
- Optimal schedules with conflict analysis for GWO.

## Requirements
- Python 3.8+
- Libraries: NumPy, Matplotlib, Random, itertools

## Contributing
Feel free to submit issues or contribute to improving the implementations.

## License
This project is licensed under the MIT License.
