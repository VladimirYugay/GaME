#!/bin/bash
#SBATCH --output=logs/%A_%a.log
#SBATCH --error=logs/%A_%a.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --array=0-2

dataset="tum"
scenes=("desk" "office" "xyz")

# Root directory containing all TUM sequence folders
TUM_DATA_PATH="<your_data_path>/TUM_RGBD"
OUTPUT_PATH="<your_output_path>"
CONFIG_PATH="configs/${dataset}"

EXPERIMENT_NAME="baseline"
SCENE_NAME=${scenes[$SLURM_ARRAY_TASK_ID]}

# Map scene names to their TUM sequence folder names
case "$SCENE_NAME" in
  desk)   SCENE_DATA_PATH="${TUM_DATA_PATH}/rgbd_dataset_freiburg1_desk" ;;
  office) SCENE_DATA_PATH="${TUM_DATA_PATH}/rgbd_dataset_freiburg3_long_office_household" ;;
  xyz)    SCENE_DATA_PATH="${TUM_DATA_PATH}/rgbd_dataset_freiburg2_xyz" ;;
esac

source /home/<your_username>/anaconda3/bin/activate
conda activate game

echo "Job for dataset: $dataset, scene: $SCENE_NAME, experiment: $EXPERIMENT_NAME"
echo "Starting on: $(date)"
echo "Running on node: $(hostname)"


python run.py --config_path "${CONFIG_PATH}/${SCENE_NAME}.yaml" \
                   --data_path "${SCENE_DATA_PATH}" \
                   --output_path "${OUTPUT_PATH}/${dataset}/${EXPERIMENT_NAME}/${SCENE_NAME}" \

echo "Job for scene $SCENE_NAME completed."
