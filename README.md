# ViT-ReCo: minimal experiment code

Reproduce the saved Imagenette-160 experiment for DeiT-Tiny, DeiT-Small, and Swin-Tiny. The model uses a fixed 10% MLP magnitude mask, two FP32 recovery epochs, two weight-only INT8 QAT epochs, group-32 coding, and Zstandard compression. The original 1,000-class heads are retained. This is **checkpoint compression**; inference reconstructs dense weights.

## Quick test (no dataset or model download)

Use Python 3.12 and run `pip install numpy zstandard`, then `python smoke_test.py`. It builds a tiny synthetic archive and checks its byte-aligned reconstruction. This is a format smoke test, **not an accuracy experiment**. For the full experiment, install a compatible PyTorch and torchvision pair and run `pip install -r requirements.txt`.

## Full experiment

Run the following commands from this directory:

```bash
OPENBLAS_NUM_THREADS=1 python resume_vitreco.py --model tiny
OPENBLAS_NUM_THREADS=1 python resume_vitreco.py --model small
OPENBLAS_NUM_THREADS=1 python resume_vitreco.py --model swin
OPENBLAS_NUM_THREADS=1 python summarize_validation.py
OPENBLAS_NUM_THREADS=1 python benchmark_loading.py --reps 5
```

The first three runs download Imagenette-160 and the public `timm` pretrained weights, check recorded hashes, train matched controls, and evaluate all 3,925 validation images. The script resumes completed stages. It saves compressed weight archives, logits, training histories, and protocols under `vitreco_validation/results/`. Training is CPU intensive. The loading benchmark measures warm-cache file read, decode, model construction, and state installation; it does not measure inference speed.

If you already have the three saved validation checkpoint ZIPs, place them beside `verify_saved.py`, install NumPy and SciPy, and run `python verify_saved.py` to audit SHA-256, 3,925-image top-1, paired tests, and archive bytes without retraining. Extract all three ZIPs into this directory to run `summarize_validation.py` and `benchmark_loading.py` on the saved outputs. The ZIPs are distributed separately because they are 93–164 MB each.

Observed ViT-ReCo archives are 1.90–2.35% smaller than matched unpruned QAT; accuracy differences are small and statistically inconclusive with one training seed. The 250 earlier pilot images are included in the 3,925-image validation, and the saved analysis also reports the remaining 3,675 separately. The result does not establish a new or generally superior compression algorithm.

