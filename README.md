# Handling Uncertain Labels in Chest X-Rays

A research project comparing strategies for handling uncertain labels in the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) chest X-ray dataset.

## Overview

Medical imaging datasets frequently contain uncertain or ambiguous labels due to radiologist disagreement or vague radiology reports. This project trains DenseNet-121 models under six different uncertainty-handling strategies and evaluates which yields the safest clinical model across five pathologies:

- **Atelectasis**
- **Cardiomegaly**
- **Consolidation**
- **Edema**
- **Pleural Effusion**

## Uncertainty Strategies

| Strategy | Description |
|---|---|
| U-Ignore | Mask uncertain labels out of the loss |
| U-Zeroes | Treat uncertain as negative |
| U-Ones | Treat uncertain as positive |
| U-SelfTrained | Train with U-Ignore, relabel uncertain samples with model predictions, retrain |
| U-MultiClass | Treat uncertainty as its own class (3-class per pathology) |
| Label Smoothing | Map uncertain to soft positive (0.55), apply smoothing globally |

## Project Structure

```
├── configs/
│   └── default.yaml          # Hyperparameters and paths
├── src/
│   ├── dataset.py            # CheXpert PyTorch Dataset
│   ├── model.py              # DenseNet-121 wrapper
│   ├── strategies.py         # Uncertainty label-mapping logic
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Metrics and plotting
│   └── utils.py              # Config, logging, checkpointing
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── checkpoints/              # Saved model weights
├── requirements.txt
└── README.md
```

## Setup

### 1. Dataset

Register at [Stanford AIMI](https://stanfordaimi.azurewebsites.net/) and download **CheXpert-v1.0-small**. Place the dataset so that `train.csv` and `valid.csv` are accessible at the path specified in `configs/default.yaml`.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Google Colab

The full CheXpert dataset is ~440 GB, but the **CheXpert-v1.0-small** release (~11 GB, 320 px JPEGs) is all this project needs and fits comfortably on Colab.

**Recommended Drive layout:**

```
MyDrive/
├── CheXpert-v1.0-small.zip      # upload the zip, not the extracted folder
└── Research Project/            # this repo
```

**Workflow:**

1. Open `notebooks/02_training.ipynb` in Colab and select **Runtime → Change runtime type → GPU** (T4 is fine on free tier).
2. Run the setup cells. The notebook will:
   - mount Drive,
   - unzip `CheXpert-v1.0-small.zip` to `/content/data/` (local SSD, ~100× faster than reading from Drive),
   - use a 5% random subset of training data by default.
3. Once the pipeline runs end-to-end on the subset, set `SUBSET_FRAC = None` in cell 3 and run a full training session.

**Tips for staying within Colab's limits:**

- Keep `image_size: 224` and `batch_size: 32` — these fit on a T4 with mixed precision.
- If Colab disconnects mid-run, comment out already-finished strategies in cell 9 and re-run.
- Checkpoints save to Drive (`checkpointing.save_dir`), so progress survives disconnects.
- If you hit GPU OOM, lower `batch_size` to 16 in `configs/default.yaml`.

## References

- Irvin, J., Rajpurkar, P., et al. "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison." *AAAI 2019*.
