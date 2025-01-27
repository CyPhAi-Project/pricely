# Instructions for HSCC 2025 Repeatability Package

Here we provide the instructions to reproduce the experiment result for our paper,
"Certifying Lyapunov Stability of Black-Box Nonlinear Systems via Counterexample Guided Synthesis",
accepted by HSCC 2025.
We focus on reproducing the following tables and figures in our submission:

+ Table 2 showing the result for the `Trans` benchmarks.
+ Table 3 showing the result for the `Polys` benchmarks.
+ Figure 3 showing the plots including the phase portrait, the basin of attraction, and the triangulation for a single benchmark, and we use the Van der Pol as an example.


## Obtain Docker Image and Create Docker Container

We provide a compressed file `hscc2025_latest.tar.gz` on [Google Drive] for loading our docker image.
Once the file is downloaded, load the image using the following command:

[Google Drive]: https://drive.google.com/file/d/1EfmdD7c0P9TCkyxo3XpP81Nf9EDlYxSA/view?usp=sharing


```shell
docker load --input hscc2025_latest.tar.gz
```

Once our docker image is successfully loaded,
use the following command to create a temporary container and log into the container:
```shell
docker run --rm -it hscc2025:latest
```
You should have successfully logged into the container as the user named `app`.
If you are using macOS with an arm64 core, you may try the following command to specify and emulate the platform:
```shell
docker run --rm -v /tmp/out:/tmp/out --platform=linux/amd64 -it hscc2025:latest
```

**Note:** Using the above command, the container will be removed automatically once your logout, and any change inside the container is discarded.


## Reproduce Experiment Result in Table 2 and 3 inside Docker Container

### Reproduce Table 2

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

The experiment result in the CSV file differs slightly from Table 2 due to bug fixes and updates after submission.
We also include a CSV file named `hscc2025_trans.expected.csv` for reference which contains the latest results collected under our setup.


### Reproduce Table 3

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

**Note:** Due to the excessive execution time (>4 hrs) for the `poly_1` benchmark,
we commented it out in `run_hscc2025_polys.py`.
You can uncomment it to run the experiment.


## Experiment and Visualization for Single Benchmark

To run the experiment for a single benchmark and generate visualization,
we provide the `run.py` script for finer configurations.
Here we follow our paper to study the Van der Pol oscillator.
To reproduce two plots in Fig. 3 and an extra plot visualizing the diameters of regions,
run the following commands:
```shell
cd ~/pricely-repo
python3 run.py
```
The script may take 1~2 mins to execute,
and you should see command line messages showing intermediate results.
After the Python script finishes successfully,
you should see the following three PNG images under the `out/<yyyy-mm-dd>/neurips2022_van_der_pol` folder:

+ `phase_portrait.png` plots the phase portrait and the basic of attraction as shown in Fig.3 (*Right*).
+ `cegus-valid_regions.png` plots the final triangulation as shown in Fig.3 (*Right*).
+ `diameters.png` plots the diameters of simplices in the triangulation in descending order.

To run the experiment for other benchmarks,
import another benchmark under the `examples` folder as a Python module.

**NOTE:** Plotting the triangulation and phase portrait is supported for only 2D dynamical systems.


## System Requirements

We have tested our docker image on the following combinations of platforms:

+ Docker Desktop 4.37.1 for Windows and Ubuntu 20.04 through Windows Subsystem for Linux
+ Docker 24.0.7 and Ubuntu 22.04
+ Docker Desktop 4.37.2 (179585) for Mac with Docker Engine 27.4.0
