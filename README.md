# Metaheuristics-Mammaprint

A comprehensive Python framework for feature selection using metaheuristic algorithms applied to the Mammaprint cancer classification dataset.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running the Pipelines](#running-the-pipelines)
  - [Pipeline 1: Feature Selection with Metaheuristics](#pipeline-1-feature-selection-with-metaheuristics)
  - [Pipeline 2: Classifier Evaluation](#pipeline-2-classifier-evaluation)
- [Configuration Parameters](#configuration-parameters)
- [Output Files](#output-files)
- [Metaheuristic Algorithms](#metaheuristic-algorithms)
- [Objective Functions](#objective-functions)
- [Classifiers](#classifiers)
- [Examples](#examples)
- [License](#license)

## Overview

This project implements multiple metaheuristic optimization algorithms for feature selection in cancer classification tasks. The framework includes two main pipelines:

1. **Feature Selection Pipeline** (`pipeline.py`): Runs metaheuristic algorithms to find optimal feature subsets
2. **Classifier Evaluation Pipeline** (`pipeline_classifiers.py`): Evaluates the selected features using various machine learning classifiers

## Features

- **6 Metaheuristic Algorithms**: Genetic Algorithm (GA), Differential Evolution (DE), Simulated Annealing (SA), Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO), Gray Wolf Optimizer (GWO)
- **6 Objective Functions**: ReliefF, Mutual Information, Symmetrical Uncertainty, Gain Ratio, Information Gain, Chi-Square
- **8 Machine Learning Classifiers**: KNN, SVM, Naive Bayes, CART, Random Forest, MLP, LDA, QDA
- **Parallel Processing**: Multi-executor support for faster computation
- **JVM Management**: Automatic Weka JVM reload for long-running experiments
- **Comprehensive Logging**: Detailed history tracking and CSV report generation

## Installation

### Prerequisites

- Python 3.8 or higher
- Java Runtime Environment (JRE) for Weka integration

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ASP-ML/Metaheuristics-Mammaprint.git
cd Metaheuristics-Mammaprint
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Verify Weka installation:
```bash
python -c "import weka.core.jvm as jvm; jvm.start(); print('Weka OK'); jvm.stop()"
```

## Project Structure

```
Metaheuristics-Mammaprint/
├── metaheuristics/          # Metaheuristic algorithm implementations
│   ├── base.py             # Base class for all algorithms
│   ├── genetic_algorithm.py
│   ├── differential_evolution.py
│   ├── simulated_annealing.py
│   ├── pso.py
│   ├── ant_colony.py
│   └── gray_wolf.py
├── utils/                   # Utility functions
│   ├── helper_functions.py # Data processing utilities
│   └── experiment.py       # Experiment management classes
├── data/                    # Dataset files
│   ├── train_normal_rtf.csv
│   ├── test_normal_rtf.csv
│   └── normal_rtf.csv
├── ofunctions.py           # Objective functions for feature selection
├── classifiers.py          # Machine learning classifier implementations
├── pipeline.py             # Main feature selection pipeline
├── pipeline_classifiers.py # Classifier evaluation pipeline
├── shared_imports.py       # Centralized imports
└── requirements.txt        # Project dependencies
```

## Running the Pipelines

### Pipeline 1: Feature Selection with Metaheuristics

The main pipeline (`pipeline.py`) runs metaheuristic algorithms to find optimal feature subsets.

#### Basic Usage

```bash
python pipeline.py
```

#### Configuration

Edit the `__main__` section in `pipeline.py` to customize parameters:

```python
if __name__ == "__main__":
    # Dataset configuration
    data, labels = processDataSet('train_normal_rtf.csv')
    
    # Experiment parameters
    STEPS_EXPERIMENT = 1        # Number of experimental steps/iterations
    START_GEN = 0               # Starting generation (usually 0)
    END_GEN = 100               # Maximum generations
    STEP_GEN = 10               # Generation increment per iteration
    NUMBER_EXECUTORS = 4        # Parallel executors (adjust based on CPU cores)
    
    # Algorithms to evaluate
    METAS_EXPERIMENT = ['GA', 'SA', 'DE']  # Available: 'GA', 'SA', 'DE', 'PSO', 'ACO', 'GWO'
    
    # Output directory configuration
    _date = "25_03_2024"        # Date folder name
    _time = "12_00_00"          # Time folder name
    base_path, historic_path = make_experiment_dirs(_date, _time, auto_time_stamp=False)
    
    # Initialize and run experiment
    experiment = Experiment(STEPS_EXPERIMENT, len(METAS_EXPERIMENT))
    mh_dict = getMetasByObjectiveFunction(
        data, labels, 
        METAS_EXPERIMENT, 
        OBJECTIVE_FUNCTIONS, 
        START_GEN + STEP_GEN, 
        path_report=historic_path
    )
    
    # Execute the pipeline
    report_paths = run_experiment_using_weka(
        mh_dict, data, labels, 
        STEPS_EXPERIMENT, 
        START_GEN + STEP_GEN, 
        END_GEN, 
        STEP_GEN, 
        NUMBER_EXECUTORS, 
        METAS_EXPERIMENT, 
        OBJECTIVE_FUNCTIONS, 
        base_path, 
        historic_path
    )
```

#### Pipeline Workflow

1. **Initialization**: Loads dataset and creates experiment directories
2. **Meta-Dictionary Creation**: Generates all algorithm-objective function combinations
3. **Parallel Execution**: Distributes computations across multiple executors
4. **JVM Management**: Automatically reloads Weka JVM when needed (every 50 iterations)
5. **Result Collection**: Aggregates results from all executions
6. **Report Generation**: Creates CSV files with feature subsets and fitness scores

#### Output Structure

```
results/
└── {date}/
    └── {time}/
        ├── total_history_1_report.csv    # Complete history for step 1
        ├── step_1_report.csv             # Best results by objective function
        └── historic/
            ├── GA_relieff.csv            # Individual algorithm history
            ├── DE_mi.csv
            └── ...
```

#### Output File Columns

**total_history_N_report.csv**:
- `Meta`: Metaheuristic algorithm name
- `OF`: Objective function name
- `IT`: Number of iterations/generations
- `Subset`: Selected feature indices
- `Length`: Number of features selected
- `Settings`: Algorithm hyperparameters
- `BestScore`: Best fitness value achieved

### Pipeline 2: Classifier Evaluation

The classifier pipeline (`pipeline_classifiers.py`) evaluates the feature subsets found by metaheuristics using various ML classifiers.

#### Basic Usage

```bash
python pipeline_classifiers.py
```

#### Configuration

Edit the `__main__` section in `pipeline_classifiers.py`:

```python
if __name__ == "__main__":
    # Path to feature selection results
    base_path = 'results/25_03_2024/12_00_00'
    last_report_path = f'{base_path}/total_history_1_report.csv'
    
    # Run classifier evaluation
    classifiers_evaluation(
        resume_file_path=last_report_path,
        data_file_name='train_normal_rtf',  # Dataset name (without .csv)
        parent_folder='25_03_2024',          # Should match pipeline.py date
        child_folder='12_00_00',             # Should match pipeline.py time
        STEP_NAME='1'                        # Step identifier for output files
    )
```

#### Pipeline Workflow

1. **Load Results**: Reads feature subsets from `pipeline.py` output
2. **Data Loading**: Loads the original dataset
3. **Feature Extraction**: Creates subsets based on selected feature indices
4. **Classifier Training**: Trains multiple classifiers on each subset
5. **Cross-Validation**: Performs stratified k-fold cross-validation
6. **Metrics Collection**: Calculates performance metrics (ACC, AUC, F1, etc.)
7. **Report Generation**: Creates detailed CSV reports

#### Output Structure

```
classifiers_results/
└── train_normal_rtf/
    └── {date}/
        └── {time}/
            ├── class_step_1_report_train_normal_rtf.csv    # Summary metrics
            ├── matrix_step_1_report_train_normal_rtf.csv   # Confusion matrices
            ├── min_len_last_step_full_history.csv          # Fold-by-fold results
            └── {algorithm}/
                └── {index}_{algorithm}_{of}_weights_{of}/  # MLP model weights
```

#### Output File Columns

**class_step_N_report.csv** (Performance Metrics):
- `Meta`: Algorithm name
- `OF`: Objective function
- `IT`: Iterations
- `Subset`: Feature indices
- `Length`: Number of features
- `Settings`: Algorithm parameters
- `Classifier`: Classifier name
- `ACC_MEAN/DESV`: Accuracy mean/standard deviation
- `PRE_MEAN/DESV`: Precision mean/std
- `REC_MEAN/DESV`: Recall mean/std
- `AUC_MEAN/DESV`: AUC-ROC mean/std
- `F1_MEAN/DESV`: F1-score mean/std
- `MCC_MEAN/DESV`: Matthews Correlation Coefficient mean/std
- `SPEC_MEAN/DESV`: Specificity mean/std

**matrix_step_N_report.csv** (Confusion Matrices):
- `tn_mean/desv/sum`: True negatives
- `fp_mean/desv/sum`: False positives
- `fn_mean/desv/sum`: False negatives
- `tp_mean/desv/sum`: True positives

**min_len_last_step_full_history.csv** (Detailed Results):
- Individual fold results for each classifier evaluation
- Complete metrics per fold for reproducibility analysis

## Configuration Parameters

### Metaheuristic Algorithm Settings

Each algorithm has specific hyperparameters that can be tuned:

#### Genetic Algorithm (GA)
- `population_size`: 50-200 (number of chromosomes)
- `generations`: 50-500 (number of evolutionary cycles)
- `mutationRange`: 0.01-0.1 (probability of bit-flip mutation)
- `n_parents`: population_size // 2 (parents for reproduction)

#### Differential Evolution (DE)
- `pop_size`: 30-100 (population size)
- `epochs`: 50-500 (number of generations)
- `mutationFactor`: 0.4-1.0 (differential weight F)
- `crossOverFactor`: 0.1-0.9 (crossover probability CR)

#### Simulated Annealing (SA)
- `initTemp`: 100-1000 (initial temperature)
- `iterations`: 100-1000 (cooling iterations)
- `alpha`: 0.85-0.99 (cooling rate)
- `beta`: 1-10 (Boltzmann constant)

#### Particle Swarm Optimization (PSO)
- `n_particles`: 20-50 (swarm size)
- `max_iters`: 50-500 (iterations)
- `w`: 0.4-0.9 (inertia weight)
- `c1`, `c2`: 1.5-2.5 (cognitive/social coefficients)

#### Ant Colony Optimization (ACO)
- `n_ants`: 10-50 (colony size)
- `iterations`: 50-300
- `alpha`: 1-2 (pheromone importance)
- `beta`: 2-5 (heuristic importance)
- `rho`: 0.1-0.5 (evaporation rate)

#### Gray Wolf Optimizer (GWO)
- `search_agents`: 5-30 (pack size)
- `max_iter`: 50-500 (iterations)

### Computational Resources

**NUMBER_EXECUTORS**: Adjust based on your system:
- 4-6 cores: `NUMBER_EXECUTORS = 2-3`
- 8-12 cores: `NUMBER_EXECUTORS = 4-6`
- 16+ cores: `NUMBER_EXECUTORS = 6-10`

**Memory Considerations**:
- Each executor uses ~2-4 GB RAM
- Weka JVM adds ~1-2 GB per executor
- Recommended: 8 GB RAM minimum for 2-3 executors

## Metaheuristic Algorithms

### Genetic Algorithm (GA)
- **Inspired by**: Natural evolution and selection
- **Operators**: Tournament selection, crossover, mutation
- **Best for**: Balanced exploration/exploitation
- **Typical settings**: 100 population, 200 generations

### Differential Evolution (DE)
- **Inspired by**: Differential mutation
- **Operators**: Mutation, crossover, selection
- **Best for**: Continuous optimization adapted to binary
- **Typical settings**: 50 population, 150 generations

### Simulated Annealing (SA)
- **Inspired by**: Metal annealing process
- **Operators**: Temperature-based acceptance
- **Best for**: Escaping local optima
- **Typical settings**: T₀=1000, α=0.95, 500 iterations

### Particle Swarm Optimization (PSO)
- **Inspired by**: Bird flocking behavior
- **Operators**: Velocity and position updates
- **Best for**: Fast convergence
- **Typical settings**: 30 particles, 200 iterations

### Ant Colony Optimization (ACO)
- **Inspired by**: Ant foraging behavior
- **Operators**: Pheromone updates
- **Best for**: Discrete optimization problems
- **Typical settings**: 20 ants, 150 iterations

### Gray Wolf Optimizer (GWO)
- **Inspired by**: Wolf pack hunting strategy
- **Operators**: Hierarchical leadership
- **Best for**: Simple implementation, good results
- **Typical settings**: 10 agents, 200 iterations

## Objective Functions

### ReliefF
- Estimates feature quality based on nearest neighbors
- Handles multi-class problems well
- Range: [-1, 1], higher is better

### Mutual Information (MI)
- Measures statistical dependence between features and target
- Information-theoretic approach
- Range: [0, ∞), higher is better

### Symmetrical Uncertainty (SU)
- Normalized mutual information
- Compensates for MI's bias toward high-cardinality features
- Range: [0, 1], higher is better

### Gain Ratio
- Information gain normalized by feature entropy
- Reduces bias toward features with many values
- Range: [0, 1], higher is better

### Information Gain (IG)
- Reduction in entropy after observing feature
- Classical feature selection metric
- Range: [0, ∞), higher is better

### Chi-Square (χ²)
- Statistical test for feature-target independence
- Fast computation
- Range: [0, ∞), higher is better

## Classifiers

### K-Nearest Neighbors (KNN)
- **Type**: Instance-based learning
- **Parameters**: k=7 (number of neighbors)
- **Pros**: Simple, no training phase
- **Cons**: Slow prediction, sensitive to feature scaling

### Support Vector Machine (SVM)
- **Type**: Kernel-based classifier
- **Parameters**: RBF kernel, C=1.0
- **Pros**: Effective in high dimensions
- **Cons**: Computationally intensive for large datasets

### Naive Bayes (NB)
- **Type**: Probabilistic classifier
- **Parameters**: Gaussian distribution
- **Pros**: Fast, works well with small datasets
- **Cons**: Assumes feature independence

### CART (Decision Tree)
- **Type**: Tree-based classifier
- **Parameters**: Gini impurity
- **Pros**: Interpretable, handles non-linear relationships
- **Cons**: Prone to overfitting

### Random Forest (RF)
- **Type**: Ensemble of decision trees
- **Parameters**: n_estimators=100
- **Pros**: Robust, reduces overfitting
- **Cons**: Less interpretable than single tree

### Multi-Layer Perceptron (MLP)
- **Type**: Neural network
- **Parameters**: 2 hidden layers, Adam optimizer
- **Pros**: Captures complex patterns
- **Cons**: Requires more data, longer training

### Linear Discriminant Analysis (LDA)
- **Type**: Linear classifier
- **Parameters**: Default
- **Pros**: Fast, works well when classes are separable
- **Cons**: Assumes Gaussian distributions

### Quadratic Discriminant Analysis (QDA)
- **Type**: Non-linear extension of LDA
- **Parameters**: Default
- **Pros**: More flexible than LDA
- **Cons**: Requires more samples per class

## Examples

### Example 1: Quick Feature Selection

```python
from utils import processDataSet, make_experiment_dirs, getMetasByObjectiveFunction, Experiment
import ofunctions

# Load data
data, labels = processDataSet('train_normal_rtf.csv')

# Quick experiment: 1 algorithm, 1 objective function, 50 generations
METAS_EXPERIMENT = ['GA']
OBJECTIVE_FUNCTIONS = ['mi']
base_path, historic_path = make_experiment_dirs(auto_time_stamp=True)

experiment = Experiment(1, 1)
mh_dict = getMetasByObjectiveFunction(
    data, labels, METAS_EXPERIMENT, OBJECTIVE_FUNCTIONS, 50, path_report=historic_path
)

# Run and get results
from pipeline import run_experiment_using_weka
reports = run_experiment_using_weka(
    mh_dict, data, labels, 1, 10, 50, 10, 2, METAS_EXPERIMENT, OBJECTIVE_FUNCTIONS, base_path, historic_path
)
print(f"Results saved to: {reports}")
```

### Example 2: Full Experiment Pipeline

```python
# Step 1: Feature Selection (pipeline.py)
# Configure and run as shown above with multiple algorithms and objective functions

# Step 2: Wait for completion, then evaluate classifiers
from pipeline_classifiers import classifiers_evaluation

# Evaluate the best feature subsets
classifiers_evaluation(
    resume_file_path='results/2024_03_25/12_00_00/total_history_1_report.csv',
    data_file_name='train_normal_rtf',
    parent_folder='2024_03_25',
    child_folder='12_00_00',
    STEP_NAME='1'
)
```

### Example 3: Custom Objective Function

```python
import ofunctions
import numpy as np

# Use specific objective function with custom parameters
def custom_fitness(positions, data, labels):
    features = np.where(positions >= 0.5)[0]
    if len(features) == 0:
        return 0
    
    # Your custom evaluation logic
    mi_score = ofunctions.mi_sklearn(features, data, labels)
    return mi_score

# Integrate into pipeline
# Replace objectiveFunction parameter with custom_fitness
```

## Troubleshooting

### Common Issues

**1. Weka JVM Error**
```
Error: JVM is not running
Solution: Ensure Java is installed and JAVA_HOME is set correctly
```

**2. Out of Memory**
```
Solution: Reduce NUMBER_EXECUTORS or increase system RAM
```

**3. Import Errors**
```
Solution: Install all requirements: pip install -r requirements.txt
```

**4. Dataset Not Found**
```
Solution: Ensure CSV files are in the data/ directory
```

**5. No Results Generated**
```
Solution: Check that START_GEN < END_GEN and STEP_GEN divides evenly
```

## Performance Tips

1. **Parallel Execution**: Use NUMBER_EXECUTORS = number of CPU cores - 1
2. **Incremental Generations**: Use smaller STEP_GEN (10-20) for checkpointing
3. **Algorithm Selection**: Start with GA and DE for best balance
4. **Objective Functions**: MI and ReliefF typically perform well
5. **Memory Management**: Monitor RAM usage with task manager

**Last Updated**: November 2025
