#!/bin/bash

#SBATCH --partition="devel"
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --cpus-per-task=24

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

echo "IDs of GPUs available: $CUDA_VISIBLE_DEVICES"
echo "No of GPUs available: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "No of CPUs available: $SLURM_CPUS_PER_TASK"
echo "nproc output: $(nproc)"

nvidia-smi

sleep 10

# Unique Job ID, either the Slurm job ID or Slurm array ID and task ID when an
# array job
if [ "$SLURM_ARRAY_JOB_ID" ]; then
    job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    job_id="$SLURM_JOB_ID"
fi

# Set user ID and name of project
repo="SynthPar"
cfg_file="IndianFemale.yaml"
#cfg_file="BlackFemale.yaml"
#cfg_file="AsianFemale.yaml"
#cfg_file="AsianMale.yaml"
#cfg_file="WhiteMale.yaml"
#cfg_file="IndianMale.yaml"
#cfg_file="BlackMale.yaml"
#cfg_file="WhiteFemale.yaml"

# print out config file
cat config/${cfg_file}

# Path to scratch directory on host
scratch_host="/raid/local_scratch"
scratch_root="$scratch_host/${USER}/${job_id}"

# Files and directories to copy to scratch before the job
inputs="."
# File and directories to copy from scratch after the job
outputs="generated_images"

# Conda environment name
env_name="synthpar"
# Conda environment YAML file
env_file="environment.yml"

# Purge modules and load conda
module purge
module load python/anaconda3

# Check if the conda environment already exists
if conda env list | grep -q "$env_name"; then
    echo "Conda environment $env_name already exists."
else
    echo "Creating conda environment $env_name from $env_file..."
    conda env create -n "$env_name" -f "$env_file"
fi

# Activate the conda environment
echo "Activating conda environment $env_name..."
source activate "$env_name"

# Command to execute
run_command="python generation_script.py -c ${cfg_file}"

##########
# Set up scratch
##########
# Copy inputs to scratch
mkdir -p "$scratch_root"
for item in $inputs; do
    echo "Copying $item to scratch_root"
    cp -r "$item" "$scratch_root"
done

##########
# Monitor and run job
##########
# Monitor GPU usage
nvidia-smi dmon -o TD -s um -d 1 > "dmon_$job_id".txt &
gpu_watch_pid=$!

# run the application
start_time=$(date -Is --utc)
$run_command
end_time=$(date -Is --utc)

# Stop GPU monitoring
kill $gpu_watch_pid

##########
# Copy outputs
##########
# Copy output from scratch_root
for item in $outputs; do
    echo "Copying $item from scratch_root"
    cp -r "$scratch_root/$item" ./
done

# Clean up scratch_root directory
rm -rf "$scratch_root"

# Deactivate the conda environment
echo "Deactivating conda environment $env_name..."
conda deactivate
