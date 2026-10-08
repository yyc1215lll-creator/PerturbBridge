# PerturbBridge

Diffusion-bridge training for cellular perturbation modeling on **BBBC021**,
**JUMP-small**, and **Allen**. The repository includes the models, training
objectives, samplers, and default configurations.

## Installation

Use Python 3.10 or 3.11 on a Linux machine with NVIDIA GPUs.

```bash
git clone https://github.com/yyc1215lll-creator/PerturbBridge.git
cd PerturbBridge
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data and weights

Prepare CellFlux-format cell images, metadata, and condition embeddings:

- **BBBC021 / JUMP-small:** see the [CellFlux data instructions](https://github.com/yuhui-zh15/CellFlux),
  [preprocessed image data](https://zenodo.org/records/8307629), and
  [metadata and embeddings](https://huggingface.co/suyc21/CellFlux).
- **Allen:** obtain the drug-perturbation data from
  [Allen Cell Explorer](https://www.allencell.org/data-downloading.html)
  and prepare the cell arrays and metadata described in [DATA.md](DATA.md).

The loaders expect preprocessed 96 × 96 cell images, not raw microscopy volumes.
Segmentation, split construction, and embedding generation are not included.
See [DATA.md](DATA.md) for the exact file schemas and image layouts.

```text
data/
├── bbbc/
│   ├── images/
│   ├── metadata.csv
│   └── embeddings.csv
├── jump/
│   ├── images/
│   ├── metadata.csv
│   └── embeddings.csv
└── allen/
    ├── images/
    ├── metadata.csv
    └── embeddings.csv
```

Download the frozen Inception-v3 feature extractor used by the auxiliary losses:

```bash
mkdir -p external
curl -L --fail \
  https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth \
  -o external/inception_v3.pth
```

Use the **torch-fidelity** weights above, not the torchvision classification
checkpoint. The bridge models train from scratch; no pretrained generator is required.

## Training

For each dataset, first prepare the frozen training-data feature statistics,
then start training. The controller runs the configured stages in sequence.

```bash
# BBBC021
python train.py --dataset bbbc --prepare
python train.py --dataset bbbc

# JUMP-small
python train.py --dataset jump --prepare
python train.py --dataset jump

# Allen
python train.py --dataset allen --prepare
python train.py --dataset allen
```

Run inside an existing GPU allocation. BBBC021 requires at least **4 GPUs**;
JUMP-small and Allen require **8 GPUs** to run all stages. The launcher selects
the stage-specific GPU counts from [configs/plans.json](configs/plans.json).

To use other storage locations:

```bash
python train.py --dataset bbbc \
  --data-root ./data \
  --output-root ./outputs \
  --weights ./external/inception_v3.pth
```

Pass the same paths to `--prepare`. Checkpoints are saved under
`outputs/<dataset>/seed42/phaseN/training/`. Rerunning a stage resumes its latest
complete epoch checkpoint. Use `--phase 1`, `--phase 2`, etc. to run individual
stages in order; retain the same output directory and seed between stages.

Default training and sampling parameters are in [configs/](configs/).
To inspect the launch commands without starting training:

```bash
python train.py --dataset bbbc --dry-run
python train.py --help
```

## Acknowledgments

This implementation builds on [CellFlux](https://github.com/yuhui-zh15/CellFlux)
and its diffusion-bridge utilities. Upstream licenses and source attributions
are retained in [third_party/](third_party/) and the corresponding source files.
Dataset and pretrained-weight licenses apply separately.
