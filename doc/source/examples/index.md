# Examples

This page contains examples demonstrating the capabilities of CVXPYgen for code generation with CVXPY.

## Getting Started

The main example that demonstrates the basic workflow of CVXPYgen can be found in [`main.py`](https://github.com/cvxgrp/cvxpygen/blob/master/examples/main.py). This example shows how to:

1. Define a convex optimization problem with CVXPY
2. Generate C code for the problem
3. Solve the problem using the generated code

## Jupyter Notebook Examples

The following Jupyter notebooks provide interactive examples for various optimization problems:

### Control and Planning

- **[MPC.ipynb](https://github.com/cvxgrp/cvxpygen/blob/master/examples/MPC.ipynb)** - Model Predictive Control example
- **[actuator.ipynb](https://github.com/cvxgrp/cvxpygen/blob/master/examples/actuator.ipynb)** - Actuator optimization

### Finance and Portfolio Management

- **[portfolio.ipynb](https://github.com/cvxgrp/cvxpygen/blob/master/examples/portfolio.ipynb)** - Portfolio optimization
- **[ADP.ipynb](https://github.com/cvxgrp/cvxpygen/blob/master/examples/ADP.ipynb)** - Approximate Dynamic Programming

### Energy and Resource Management

- **[charging.ipynb](https://github.com/cvxgrp/cvxpygen/blob/master/examples/charging.ipynb)** - Electric vehicle charging optimization
- **[resource.ipynb](https://github.com/cvxgrp/cvxpygen/blob/master/examples/resource.ipynb)** - Resource allocation

### Network Optimization

- **[network.ipynb](https://github.com/cvxgrp/cvxpygen/blob/master/examples/network.ipynb)** - Network flow optimization

## Research Examples

For more advanced examples used in research papers, see the `paper_grad/` directory which contains:

- **ADP.py** - Approximate Dynamic Programming with gradients
- **elasticnet.py** - Elastic net regularization
- **portfolio.py** - Portfolio optimization with gradients

## Visualization Examples

Interactive visualization examples can be found in the `visualization/` directory:

- **actuator.py** - Actuator control visualization
- **network.py** - Network optimization visualization  
- **resource.py** - Resource allocation visualization

## Running the Examples

To run any of the Jupyter notebook examples:

1. Install CVXPYgen: `pip install cvxpygen`
2. Install Jupyter: `pip install jupyter`
3. Navigate to the examples directory
4. Start Jupyter: `jupyter notebook`
5. Open any of the `.ipynb` files

For the Python script examples, simply run them with:
```bash
python main.py
```

All examples demonstrate different aspects of CVXPYgen's capabilities, from basic code generation to advanced features like gradient computation and real-time optimization.
