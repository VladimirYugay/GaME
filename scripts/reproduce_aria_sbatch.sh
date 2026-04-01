#!/bin/bash
#SBATCH --output=logs/%A_%a.log
#SBATCH --error=logs/%A_%a.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --array=0-1

dataset="aria"
scenes=("room0" "room1")
DATA_PATH="<your_data_path>/GaME_Aria"
OUTPUT_PATH="<your_output_path>"
CONFIG_PATH="configs/${dataset}"

EXPERIMENT_NAME="baseline"
SCENE_NAME=${scenes[$SLURM_ARRAY_TASK_ID]}

source /home/<your_username>/anaconda3/bin/activate
conda activate game

echo "Job for dataset: $dataset, scene: $SCENE_NAME, experiment: $EXPERIMENT_NAME"
echo "Starting on: $(date)"
echo "Running on node: $(hostname)"


python run.py --config_path "${CONFIG_PATH}/${SCENE_NAME}.yaml" \
                   --data_path "${DATA_PATH}" \
                   --output_path "${OUTPUT_PATH}/${dataset}/${EXPERIMENT_NAME}/${SCENE_NAME}" \

echo "Job for scene $SCENE_NAME completed."
