#!/bin/bash
#SBATCH --output=logs/%A_%a.log
#SBATCH --error=logs/%A_%a.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --array=0

dataset="flat"
DATA_PATH="<your_data_path>/GaME_Flat"
OUTPUT_PATH="<your_output_path>"
CONFIG_PATH="configs/${dataset}"

EXPERIMENT_NAME="baseline"

source /home/<your_username>/anaconda3/bin/activate
conda activate game

echo "Job for dataset: $dataset, experiment: $EXPERIMENT_NAME"
echo "Starting on: $(date)"
echo "Running on node: $(hostname)"


python run.py --config_path "${CONFIG_PATH}/flat.yaml" \
                   --data_path "${DATA_PATH}" \
                   --output_path "${OUTPUT_PATH}/${dataset}/${EXPERIMENT_NAME}/" \

echo "Job for scene $SCENE_NAME completed."
