<p align="center">

  <h1 align="center">GaME: Gaussian Mapping for Evolving Scenes</h1>
  <h3 align="center">CVPR 2026</h3>
  <p align="center">
    <a href="https://vladimiryugay.github.io/"><strong>Vladimir Yugay*</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=TNKymuMAAAAJ&hl=de"><strong>Kersten Thies*</strong></a>
    ·
    <a href="https://lucacarlone.mit.edu/"><strong>Luca Carlone</strong></a>
    ·
    <a href="https://staff.fnwi.uva.nl/th.gevers/"><strong>Theo Gevers</strong></a>
    ·
    <a href="https://oswaldm.github.io/"><strong>Martin Oswald</strong></a>
    ·
    <a href="https://schmluk.github.io/"><strong>Lukas Schmid</strong></a>
  </p>
  <h3 align="center"><a href="https://vladimiryugay.github.io/game">Project Page</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./assets/comparison_aria_0.gif" width="80%">
  </a>
</p>

## ⚙️ Setting Things Up

Clone the repo:

```
git clone https://github.com/VladimirYugay/GaME.git
```

We tested the installation with ```gcc``` and ```g++``` of versions 10, 11 and 12. Also, make sure that ```nvcc --version``` matches ```nvidia-smi``` version.

Run the following commands to set up the environment
```
conda create -n game python=3.10 -y
conda activate game
```
Install pytorch:
```
# CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```
Install other dependencies:
```
conda install -c conda-forge faiss-gpu=1.8.0 git-lfs
pip install ./submodules/diff-gaussian-rasterization/ --no-build-isolation
pip install ./submodules/flashsplat-rasterization/ --no-build-isolation
pip install ./submodules/simple-knn/ --no-build-isolation
pip install -r requirements.txt
```

## 🔨 Running GaME

Here we elaborate on how to load the necessary data, configure GaME for your use-case, debug it, and how to reproduce the results mentioned in the paper.

  <details>
  <summary><b>Getting the Data</b></summary>
  We tested our code on Flat and Aria datasets. Make sure to install git lfs and hugging face cli before proceeding.
  <br>
  <br>

  **Flat** was created by <a href="https://github.com/ethz-asl/panoptic_mapping">Panoptic Mapping</a> authors. However, it is a bit tricky to find it on the web. Therefore, we uploaded it to HF datasets for easier access. Install git lfs and download it by running: <br>
  <code>git lfs install</code> <br>
  <code>git clone https://huggingface.co/datasets/voviktyl/GaME_Flat</code> <br>

  **Aria** consists of clips from <a href="https://www.projectaria.com/datasets/adt/">AriaDigitalTwin</a>. The clips are covering the area that undergoes changes outside of the camera view.
  You can download already processed data with the command: <br>
  <code>git clone https://huggingface.co/datasets/voviktyl/GaME_Aria</code> <br>
  
  </details>

  <details>
  <summary><b>Running the code</b></summary>

  ```
  python run.py --config_path configs/<dataset_name>/<config_name> --data_path <path_to_the_scene> --output_path <output_path>
  ```
  For example:
  ```
  python run.py --config_path configs/AriaMultiagent/room0.yaml --data_path <path_to>/AriaMultiagent/room0 --output_path output/AriaMultiagent/room0
  ```
  Check the configs to set up wandb for loggin.
  </details>

  <details>
  <summary><b>Reproducing Results</b></summary>
  While we tried to make all parts of our code deterministic, differential rasterizer of Gaussian Splatting is not. The metrics can be slightly different from run to run.

  If you are running on a SLURM cluster, you can reproduce the results for all the datasets by running the files in the `scripts` folder. For example, for Aria dataset:
  ```
  ./scripts/reproduce_aria_sbatch.sh
  ```
  </details>

  ## 🙏 Acknowledgments

Mapping module is based on <a href="https://github.com/VladimirYugay/Gaussian-SLAM">Gaussian-SLAM</a>. We thank the authors of <a href="https://github.com/florinshen/FlashSplat">FlashSplat</a> for their work.

## 📌 Citation

If you find our paper and code useful, please cite us:

```bib
@misc{yugay2025gaussianmappingevolvingscenes,
      title={Gaussian Mapping for Evolving Scenes}, 
      author={Vladimir Yugay and Thies Kersten and Luca Carlone and Theo Gevers and Martin R. Oswald and Lukas Schmid},
      year={2025},
      eprint={2506.06909},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.06909}, 
}
```