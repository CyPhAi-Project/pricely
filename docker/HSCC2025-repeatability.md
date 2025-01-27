# Instructions for HSCC 2025 Repeatability Package

Here we provide the instructions to reproduce the experiment result for our paper,
"Certifying Lyapunov Stability of Black-Box Nonlinear Systems via Counterexample Guided Synthesis",
accepted by HSCC 2025.


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

**Note:** Using the above command, the container will be removed automatically once your logout, and any change inside the container is discarded.


## Reproduce Experiment Result in Table 2 and 3 inside Docker Container

### Reproduce Table 2

To reproduce our result in Table 2 for the `Trans` benchmarks,
run the following commands:
```shell
cd ~/pricely-repo
python3 run_hscc2025_trans.py
```
After the Python script finishes successfully,
a CSV file named `hscc2025_trans.csv` is generated under the `out/` folder.


### Reproduce Table 3

To reproduce our result in Table 3 for the `Polys` benchmarks with the radius of the region of interest X set to `r=1`,
run the following commands:
```shell
cd ~/pricely-repo
python3 run_hscc2025_polys.py
```
After the Python script finishes successfully,
a CSV file named `hscc2025_polys.csv` is generated under the `out/` folder.

To run the experiment with the radius set to `r=5` of `r=10`,
modify the constant `X_NORM_UB` in each of the benchmark file `examples/*.py`.

**Note:** Due to the excessive execution time for the `poly_1` benchmark,
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
After the Python script finishes successfully,
you should see the following three PNG images under the `out/<yyyy-mm-dd>/neurips2022_van_der_pol` folder:

+ `phase_portrait.png` plots the phase portrait and the basic of attraction in Fig.3 (*Right*).
+ `cegus-valid_regions-5x5.png` plots the final triangulation in Fig.3 (*Right*).
+ `diameters-5x5.png` plots 

**NOTE:** Plotting the triangulation and phase portrait is supported for 2D dynamical systems.
