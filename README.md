# ViT-ReCo

ViT-ReCo combines fixed 10% MLP magnitude pruning, teacher-guided recovery, weight-only INT8 quantization-aware training (QAT), and Zstandard compression for compact vision-transformer checkpoints.

This repository reproduces the Imagenette-160 experiments for **DeiT-Tiny, DeiT-Small, and Swin-Tiny**. Models retain their original 1,000-class classification heads and reconstruct dense FP32 weights for inference.

## Quick Test

With Python 3.12, run:

```bash
pip install numpy zstandard
python smoke_test.py
```

The test creates a small synthetic archive and verifies byte-aligned weight reconstruction.

## Full Experiment

Install a compatible PyTorch and torchvision pair, then the remaining dependencies:

```bash
pip install -r requirements.txt
```

Run from the repository directory:

```bash
OPENBLAS_NUM_THREADS=1 python resume_vitreco.py --model tiny
OPENBLAS_NUM_THREADS=1 python resume_vitreco.py --model small
OPENBLAS_NUM_THREADS=1 python resume_vitreco.py --model swin
OPENBLAS_NUM_THREADS=1 python summarize_validation.py
OPENBLAS_NUM_THREADS=1 python benchmark_loading.py --reps 5
```

The pipeline downloads Imagenette-160 and pretrained `timm` weights, verifies recorded hashes, trains matched controls, and evaluates all **3,925 validation images**. Completed stages are resumed automatically.

Training uses two FP32 recovery epochs followed by two QAT epochs. Weights are encoded in groups of 32 and compressed with Zstandard. Experiments run on CPU.

Outputs are saved under `vitreco_validation/results/`, including:

- Compressed model checkpoints.
- Validation logits and accuracy results.
- Training histories and experiment settings.
- Statistical comparisons and timing summaries.

The loading benchmark measures warm-cache file access, decoding, model construction, and parameter installation.

## Verify Saved Results

Place the three saved validation checkpoint ZIPs beside `verify_saved.py`, then run:

```bash
pip install numpy scipy
python verify_saved.py
```

This verifies SHA-256 hashes, validation accuracy, paired statistical tests, and archive sizes using the saved outputs.

To summarize results and benchmark loading, extract all three ZIPs into the repository directory and run:

```bash
OPENBLAS_NUM_THREADS=1 python summarize_validation.py
OPENBLAS_NUM_THREADS=1 python benchmark_loading.py --reps 5
```

Checkpoint ZIPs are distributed separately and range from approximately 93–164 MB each.

