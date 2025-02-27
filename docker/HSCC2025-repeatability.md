# Instructions for HSCC 2025 Repeatability Evaluation

In this document, we provide the instructions to reproduce the experiment result for our paper,
"Certifying Lyapunov Stability of Black-Box Nonlinear Systems via Counterexample Guided Synthesis",
accepted by the International Conference on Hybrid Systems: Computation and Control (HSCC) 2025.
We focus on reproducing the following tables and figures in our submission:

+ Table 2 showing the result for the `Trans` benchmarks.
+ Table 3 showing the result for the `Polys` benchmarks.
+ Figure 3 showing the plots including the phase portrait, the basin of attraction, and the triangulation for a single benchmark, and we use the Van der Pol as an example.

We assume the user is familar with Docker and the `bash` command line interface for Linux. 
In [Obtain Docker Image and Create Docker Container](#obtain-docker-image-and-create-docker-container), we provide minimum docker commmands for loading our docker image, creating a docker container of our customized Ubuntu Linux OS, and logging into the Ubuntu OS inside the docker container.
In the rest of the sections, we assume the user is executing commands with the `bash` command line interface in the Ubuntu OS, and all experiments result and files are generated inside the container.

The source code of our prototype tool is publicly available at https://github.com/CyPhAi-Project/pricely.


## Obtain Docker Image and Create Docker Container

We provide a compressed file `hscc2025_latest.tar.gz` on [Google Drive] for loading our docker image.

[Google Drive]: https://drive.google.com/file/d/1EfmdD7c0P9TCkyxo3XpP81Nf9EDlYxSA/view?usp=sharing

<https://drive.google.com/file/d/1EfmdD7c0P9TCkyxo3XpP81Nf9EDlYxSA/view?usp=sharing>

Once the file is downloaded, load the image using the following command:
```shell
docker load --input hscc2025_latest.tar.gz
```

Once our docker image is successfully loaded,
use the following command to create a temporary container and log into the container:
```shell
docker run --rm -it hscc2025:latest
```
You should have successfully logged into the container as the user named `app`.
If you are using macOS with an arm64 core, you may try the following command to specify and emulate the Linux platform:
```shell
docker run --rm -v /tmp/out:/tmp/out --platform=linux/amd64 -it hscc2025:latest
```

**NOTE:** Using the above command, the container will be removed automatically once your logout, and any change inside the container is discarded.


## Reproduce Experiment Result in Table 2

To reproduce our result in Table 2 for the `Trans` benchmarks,
run the following commands:
```shell
cd ~/pricely-repo
python3 run_hscc2025_trans.py
```
The script may take 5~10 mins to execute for all benchmarks,
and you should see command line messages showing intermediate results for each benchmark.
After the Python script finishes successfully,
a CSV file named `hscc2025_trans.csv` is generated under the `out/` folder.

The experiment result in the CSV file may differ slightly from Table 2 due to bug fixes and updates after submission.
We include a CSV file named `hscc2025_trans.expected.csv` for reference which contains the latest results collected under our setup.


## Reproduce Experiment Result in Table 3

To reproduce our result in Table 3 for the `Polys` benchmarks with the radius of the region of interest X set to `r=1`,
run the following commands:
```shell
cd ~/pricely-repo
python3 run_hscc2025_polys.py
```
The script may take 5~10 mins to execute for all benchmarks,
and you should see command line messages showing intermediate results for each benchmark.
After the Python script finishes successfully,
a CSV file named `hscc2025_polys.csv` is generated under the `out/` folder.

The experiment result in the CSV file differs slightly from Table 3 due to bug fixes and updates after submission.
We also include a CSV file named `hscc2025_polys.expected.csv` for reference which contains the latest results collected under our setup.

To run the experiment with the radius set to `r=5` of `r=10`,
modify the constant `X_NORM_UB` in each of the benchmark file `examples/*.py`.

**NOTE:** Due to the excessive execution time (>4 hrs) for the `poly_1` benchmark,
we commented it out in `run_hscc2025_polys.py`.
You can uncomment it to run the experiment.


## Reproduce Plots in Figure 3 / Visualization for Single Benchmark

Apart from scripts for generating tables, we provide the `run.py` script for finer configurations for analyzing a single benchmark.
Here we follow our paper to study the Van der Pol oscillator.
To reproduce the two plots in Fig. 3 and an extra plot visualizing the diameters of regions,
run the following commands:
```shell
cd ~/pricely-repo
python3 run.py
```
The script may take 1~2 mins to execute,
and you should see command line messages showing intermediate results.
After the Python script finishes successfully,
you should see the following three PNG images under the `out/<yyyy-mm-dd>/neurips2022_van_der_pol` folder:

+ `phase_portrait.png` plots the phase portrait and the basin of attraction as shown in Fig.3 (*Right*).
+ `cegus-valid_regions.png` plots the final triangulation as shown in Fig.3 (*Right*).
+ `diameters.png` plots the diameters of simplices in the triangulation in descending order.

To run the experiment for other benchmarks,
modify `run.py` to import another benchmark under the `examples` folder as a Python module.

**NOTE:** Plotting the triangulation and phase portrait is supported for only 2D dynamical systems.


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
        Takes an array of states and return the value of the derivate of states
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
        ...  # Details omitted
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
    ...    # Details omitted
    # Modify to import the new example system
    import examples.hscc2014_normalized_pendulum as mod
    ...
    ```

5. Run the modified `run.py` script to execute the new example system:
    ```sh
    python3 run.py
    ```


## System Requirements

We have tested our docker image on the following combinations of platforms:

+ Docker Desktop 4.37.1 for Windows and Ubuntu 20.04 through Windows Subsystem for Linux
+ Docker 24.0.7 and Ubuntu 22.04
+ Docker Desktop 4.37.2 (179585) for Mac with Docker Engine 27.4.0
