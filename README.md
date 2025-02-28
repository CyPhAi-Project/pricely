# Pricely: **Pr**oject on **ICE**-learning for **Ly**apunov function synthesis

## Description
This project aims to apply the [ICE-learning] framework, a formal learning framework for synthesizing invariants from program executions, to synthesize Lyapunov-like stability certificates from trajectories of hybrid systems. Specifically, we consider the system under analysis as a black-box system.

Currently, we have developed a baseline to synthesize Lyapunov functions for continuous dynamical systems, and our first paper "Certifying Lyapunov Stability of Black-Box Nonlinear Systems via Counterexample Guided Synthesis" is accepted by the International Conference on Hybrid Systems: Computation and Control (HSCC) 2025.
Extensions to hybrid systems are ongoing.

[ICE-learning]: https://doi.org/10.1007/978-3-319-08867-9_5


## Installation

### Prerequisite

Our prototype requires Python 3.10 or above and the dReal solver for SMT queries.
Please refer to https://github.com/dreal/dreal4 and install dReal beforehand.

### Installing via conda

1. Install Miniconda or Anaconda from https://docs.conda.io/en/latest/miniconda.html.
2. Clone our repository:
    ```sh
    git clone https://github.com/hc825b/pricely.git
    cd pricely
    ```

3. Create a new conda environment and install the dependencies using `environment.yml`:
    ```sh
    conda env create -f environment.yml
    conda activate pricely
    ```

4. Install our package using pip:
    ```sh
    pip install .
    ```

### Installing via pip

1. Clone our repository:
    ```sh
    git clone https://github.com/hc825b/pricely.git
    cd pricely
    ```

2. Install our package using pip:
    ```sh
    pip install .
    ```

### Installing via Docker

To install and set up the project using Docker, follow these steps:

1. Clone our repository:
    ```sh
    git clone https://github.com/hc825b/pricely.git
    cd pricely
    ```

2. Build our Docker image:
    ```sh
    docker build -t pricely:latest -f docker/Dockerfile .
    ```

3. Run the Docker container:
    ```sh
    docker run -it --rm pricely:latest
    ```


## Usage
To run the project, execute the `run.py` script under the root of the cloned repository. This script imports the example system `neurips2022_van_der_pol` from the `examples` folder. It will validate the provided Lipschitz constant(s), synthesize a Lyapunov function for the system, and validate the synthesized Lyapunov function if the white-box model is given.

Run the script:
```sh
python3 run.py
```

The script may take 1~2 mins to execute,
and you should see command line messages showing intermediate results.
After the Python script finishes successfully,
you should see the following three PNG images under the `out/<yyyy-mm-dd>/neurips2022_van_der_pol` folder:

+ `phase_portrait.png` plots the phase portrait and the basin of attraction.
+ `cegus-valid_regions.png` plots the final triangulation.
+ `diameters.png` plots the diameters of simplices in the triangulation in descending order.

To run the experiment for other benchmarks,
modify `run.py` to import another benchmark under the `examples` folder as a Python module.


### Adding a New Example System

To add a new example system, follow these steps:

1. Create a new Python file in the `examples` folder.
   For instance, see the system in `examples/hscc2014_normalized_pendulum.py`.
   We would like to define a 2D system of the following nonlinear ODEs:
   ```math
   \begin{align*} 
   \dot{x}_0 &= x_1 \\
   \dot{x}_1 &= -\sin(x_0) - x_1
   \end{align*}
   ```

2. Define the system dynamics and other required parameters in the new file. Here is the normalized pendulum system:  
    Define the state space and region of interest.
    ```python
    # System dimension
    X_DIM = 2
    # Specify the region of interest by X_NORM_LB ≤ ‖x‖ ≤ X_NORM_UB
    X_NORM_UB = 1.0
    X_NORM_LB = 0.1
    # Specify a rectangular domain covering the region of interest
    X_LIM = np.array([
        [-X_NORM_UB, -X_NORM_UB],  # Lower bounds
        [+X_NORM_UB, +X_NORM_UB]  # Upper bounds
    ])
    ```
    Define the black-box dynamical system as the function `f_bbox`.
    ```python
    def f_bbox(x: np.ndarray):
        """ Black-box system dynamics
        Takes an array of states and return the value of the derivative of states
        """
        x0, x1 = x[:, 0], x[:, 1]
        dxdt = np.zeros_like(x)
        dxdt[:, 0] = x1
        dxdt[:, 1] = -np.sin(x0) - x1
        return dxdt
    ```
    Define the function `calc_lip_bbox` to provide Lipschitz bounds for some given regions. In this example, we provide the same Lipschitz bound for all regions.
    ```python
    def calc_lip_bbox(x_regions: np.ndarray) -> np.ndarray:
        ...
        ## Use the same Lipschitz bound for all regions
        res = np.full(shape=x_regions.shape[0], fill_value=np.sqrt(3.0))
        assert res.ndim == 1 and res.shape[0] == x_regions.shape[0]
        return res
    ```
3. (Optional) Define the white-box model as `f_expr` using dReal expressions for validation.
    ```python
    def f_expr(x_vars: Sequence[Variable]) -> Sequence[Expr]:
        assert len(x_vars) == X_DIM
        x0, x1 = x_vars
        return [x1,
                -Sin(x0) - x1]
    ```

4. Modify `run.py` to import and use the new example system:

    ```python
    ...
    # Modify to import the new example system
    import examples.hscc2014_normalized_pendulum as mod
    ...
    ```

5. Run the modified `run.py` script to execute the new example system:
    ```sh
    python3 run.py
    ```

## Code Structure

The project is organized into the following main components:

### Core Implementation (`pricely/`)
- `cegus_lyapunov.py`: Main algorithm for searching Lyapunov functions
- `candidates.py`: Generation of candidate functions
- `gen_cover.py`: Cover generation algorithms
- `utils.py`: Utility functions

### Approximation Methods (`pricely/approx/`)
- `boxes.py`: Box-based approximation techniques
- `simplices.py`: Simplex-based approximation methods

### Learning Components (`pricely/learner/`)
- `cvxpy.py`: CVXPY-based optimization learner
- `mock.py`: Mock implementation for testing purposes

### Verification Tools (`pricely/verifier/`)
- `smt_dreal.py`: SMT verification using dReal solver

### Example Applications (`examples/`)
Contains various example scripts demonstrating different use cases of the library.

### Analysis and Visualization Scripts (`scripts/`)
Tools for analyzing results and visualizing data.

### Docker Setup (`docker/`)
Docker configuration for reproducible environment setup.

## License
This project is licensed under the University of Illinois/NCSA Open Source License. See the LICENSE file for details.

## Contact
For any inquiries, please contact:
- Name: Hsieh, Chiao
- Email: hsieh.chiao.7k@kyoto-u.ac.jp
- GitHub: [hc825b](https://github.com/hc825b)
