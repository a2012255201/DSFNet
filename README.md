# DSFNet

Code for **DSFNet: Directional Spatial–Frequency Collaboration Network for Underwater Optical Image Enhancement**.

DSFNet combines directional spatial feature extraction, frequency-domain processing, and spatial–frequency feature interaction for underwater image enhancement. This repository contains the model implementation, training and inference scripts, and the evaluation code used in our experiments.

## Repository structure

```text
.
├── config.yaml                  # Training configuration
├── run.py                       # Training entry point
├── predict.py                   # Image enhancement entry point
├── data/uieb.py                 # UIEB dataset loader
├── engine/dehaze.py             # Training and validation loops
├── model/arch/DSFNet.py         # Full DSFNet model
├── model/arch/                  # Ablation variants, if included
├── test_metric/test_metric.py   # Image-quality evaluation
└── requirements.txt             # Base Python dependencies
```

Run the commands below from the repository root. A CUDA-capable GPU is required by the current training and inference scripts.

## Environment

Install a CUDA-compatible version of PyTorch and torchvision for your system. Then install the base requirements and the packages imported by the training and evaluation scripts:

```bash
pip install -r requirements.txt
pip install Pillow scipy scikit-image kornia PyYAML ruamel.yaml tensorboard ptflops pandas openpyxl
```

## Data preparation

The dataset loader expects paired images with matching filenames in `input` and `gt` directories:

```text
UIEB/
├── train/
│   ├── input/
│   └── gt/
├── val/
│   ├── input/
│   └── gt/
└── test/
    └── input/
```

Set `data.train_path` to the `train` directory and `data.valid_path` to the `val` directory in `config.yaml`. Dataset images are not included in this repository.

## Training

The provided configuration trains for 600 epochs with a batch size of 8, a 256 × 256 training crop, and an initial learning rate of `2e-4`. Validation images are resized to 256 × 256. Edit the dataset paths in `config.yaml`, then run:

```bash
python run.py
```

Validation is performed every epoch. The checkpoint with the highest validation PSNR is saved to `checkpoint/DSFNet/DSFNet/ckpt/best_psnr.pth` with the provided configuration.

## Inference

```bash
python predict.py \
  checkpoint/DSFNet/DSFNet/ckpt/best_psnr.pth \
  /path/to/UIEB/test/input \
  ./results
```

The three positional arguments are the checkpoint path, input image directory, and output root directory. The script processes images at their original resolution and saves them to `<output_root>/<input_directory_name>/`. For the example above, the output directory is `./results/input/`.

## Evaluation

The metric script is `test_metric/test_metric.py`. Before running it, edit `prd_root`, `gt_root`, and `xlsx_name` in its `__main__` block to match your local directories. It expects predictions and reference images under its predefined dataset paths, such as `01_UIEB/01_val/` for validation predictions and `01_UIEB/01_val/gt/` under the reference root. Then run:

```bash
python test_metric/test_metric.py
```

The script reports PSNR, SSIM, and MSE for datasets configured with references, as well as UIQM and UCIQE for available prediction folders. Results are written to an Excel file in `./00_metric/` by default.

## Ablation models

If the ablation implementations are included in `model/arch/`, select a variant by changing the model import in both `run.py` and `predict.py` from `model.arch.DSFNet` to the desired module. These variants expose the model class as `DSFNet`. Train each variant separately and pass its matching checkpoint to `predict.py`.
