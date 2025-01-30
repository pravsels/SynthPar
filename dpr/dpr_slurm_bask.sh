#!/bin/bash

#SBATCH --qos=turing
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1 # change this if you want to increase the number of gpus. Available values from 1 to 4
#SBATCH --time=6-00:00:00
#SBATCH --constraint=a100_80 # if you want to use 80GB GPU, change it to a100_80
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

repo="face_editing"
subdir="techniques"
name="dpr"
other_name="dragan"

module purge 
module load baskerville 

# cfg="0.yaml"
# cfg="1.yaml"
# cfg="2.yaml"
# cfg="3.yaml"
# cfg="4.yaml"
# cfg="5.yaml"
# cfg="6.yaml"
# cfg="7.yaml"

# print out config file
cat configs/${cfg}

# Path to scratch directory on host
project_dir="/bask/projects/v/vjgo8416-tdi4/"
repo_dir="$project_dir/${repo}/${subdir}/${name}"
# Singularity container
container="../${other_name}/container/${other_name}_ws.sif"
# Singularity 'exec' command
container_command="python network_demo_512.py -c configs/${cfg}"
# Command to execute
run_command="apptainer exec
  --nv
  --pwd $repo_dir
  --env CUDA_VISIBLE_DEVICES="0"
  $container
  $container_command"

##########
# Run job
##########

# run the application
start_time=$(date -Is --utc)
$run_command
end_time=$(date -Is --utc)

# Print summary
echo "executed: $run_command"
echo "started: $start_time"
echo "finished: $end_time"

