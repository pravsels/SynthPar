#!/bin/bash

CONDA_ENV=synthpar2

conda env create -f environment.yml

# Clean up any unnecessary files
conda run -n $CONDA_ENV conda clean -y --all

