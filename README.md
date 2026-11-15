# ECE 285 Homework 2: WikiArt GAN

Code for [ECE 285 Homework 2](output/pdf/ECE285_HW2_Report_Hongyi_Ding.pdf).

This project started with a small DCGAN baseline for 128 x 128 paintings,
then compared three width-64 configurations. R1 at epoch 200 gave the best
validation FID, so its averaged generator weights (EMA) were used for test.

## Setup and data

Use Python 3.11 or newer:

```bash
pip install -r requirements.txt
python smoke_test.py --device cpu --inception
```

The experiments used PyTorch 2.7.1+cu128 and Ignite 0.5.4 on the course GPU.
The Inception check downloads ImageNet weights on its first run.
On MPS, spectral normalization needs `PYTORCH_ENABLE_MPS_FALLBACK=1`.

Data comes from [WikiArt](https://www.kaggle.com/datasets/steubk/wikiart).
A random subset of 12,000 images (seed 2026) was resized with aspect ratio
preserved and center-cropped to 128 x 128. Removing 12 duplicate images before
splitting left 9,988 training, 1,000 validation and 1,000 test images.
All 27 styles are included; the model does not use style labels.

Extract the prepared data into `data/`, containing `images/` and `splits.csv`.
To prepare a new copy from the downloaded style folders:

```bash
python data.py /path/to/wikiart --output data_new --limit 12000
```

## Experiments

The latent vector has 128 dimensions. All runs use Adam with learning rate
0.0002, betas (0.5, 0.999), batch size 64 and seed 2026. Each batch has one
discriminator update and one generator update.

| Run | Configuration | Epochs |
|---|---|---|
| Baseline | Width 32, transposed convolutions, BatchNorm, non-saturating BCE | 100 |
| R1 | Width 64, horizontal flips, generator EMA (0.999) | 200 |
| R2 | R1 + DiffAugment: color, translation, cutout | 200 |
| R3 | R2 + nearest-neighbor/conv upsampling, spectral norm in D, hinge loss | 200 |

Commands to reproduce the training runs:

```bash
python train.py --data data --device cuda --epochs 100 --output output/dcgan128
DATA=data DEVICE=cuda bash experiments.sh r1 r2 r3
```

Each run writes its settings, epoch-average losses, fixed-noise samples and
generator checkpoints. R1-R3 save checkpoints at epoch 1 and every 20 epochs,
then evaluate them on validation images. Use a new output directory when
repeating a completed run.

EMA checkpoints store averaged weights as `generator` and training weights as
`generator_raw`. Before saving, EMA BatchNorm statistics are recalculated from
2,048 noise vectors. The files contain generator weights for evaluation;
optimizer and discriminator states are not saved.

## Results

| Model | Epoch | Validation FID | IS |
|---|---|---|---|
| Baseline | 100 | 0.0950 | 2.90 |
| R1 | 200 | 0.0706 | 3.64 |
| R2 | 180 | 0.0918 | 2.81 |
| R3 | 160 | 0.0992 | 1.95 |

R1 was selected by validation FID. Its final test FID is **0.0830**, with
**IS 3.64**. At epoch 100, R1 had validation FID 0.1247; its final result also
benefits from a longer training run than the baseline.

Evaluation follows the course's
[Ignite tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/).
It compares 1,000 real and 1,000 generated images, converted to 8-bit RGB and
bilinearly resized to 299 x 299. FID uses 1,000-dimensional Inception class
probabilities. Its numerical scale is much smaller than the usual 2,048-feature
FID, and the two scores are not directly comparable. IS uses the whole generated
set as one split. Validation and
test reuse the same generated samples, so R1's IS is the same for both.

The final evaluation has already been run. Its command was:

```bash
python evaluate.py --checkpoint output/r1/epoch_200.pt --data data --split test --device cuda --output output/test_r1_final
```

`--raw` evaluates the training weights instead of EMA. `reference.py` computes
scores for real images with the same preprocessing. The CUDA reference gives
real train-vs-test FID 0.0410 and real test IS 5.51.

- Report: `output/pdf/ECE285_HW2_Report_Hongyi_Ding.pdf`
- LaTeX source and figure script: `report/`
- Final metrics: `output/test_r1_final/metrics.json`
- Model selection: `output/final_selection.json`

The repository includes the reported metrics and the files needed to rebuild
the report figures. Dataset images and model weights are stored separately.
