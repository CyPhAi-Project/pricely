# Prepare the docker image and container for repeatability

## Build the docker image
The build command below only works for **project members** with [proper setup of SSH keys on GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).
```shell
docker build --tag hscc2025 .
```

## Save the docker image as a gzip file

```shell
docker save hscc2025:latest | gzip > hscc2025_latest.tar.gz
```

## Load the docker image on another host machine
```shell
docker load --input hscc2025_latest.tar.gz
```

## Create a temporary docker container for reproducing experiment results
```shell
docker run --rm --name hscc2025 -it hscc2025:latest
```
This will create a container from the image and start a bash shell for running commands for experiments.
The container will be automatically removed after logging out.
