#!/bin/bash

# Load necessary Hyak modules
module load cuda/11.8.0

# Load hoomd by activating the python virtual environment in which HOOMD was built.
# Different file path needed for different users.
source /gscratch/zeelab/zsherm/hoomd/hoomd2.9.7-cuda11.8.0/hoomd-venv/bin/activate

# Add the amgx libraries to the file path. Should consider doing this automatically during install?
export LD_LIBRARY_PATH=/gscratch/zeelab/zsherm/amgx/amgx2.3.0-cuda11.8.0/AMGX/build:$LD_LIBRARY_PATH

# Run simulation
python run.py
