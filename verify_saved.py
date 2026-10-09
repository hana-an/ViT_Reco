"""Audit the saved ViT-ReCo validation archives without retraining."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json

import numpy as np
from scipy.stats import binomtest


EXPECTED_SHA256 = {
    "ViT_ReCo_tiny_Validation_Checkpoint.zip": "07e629389b69c75058a68064fedc39ac895cd0937fd1a0c04ffc4f438334ad14",
    "ViT_ReCo_small_Validation_Checkpoint.zip": "f799fdee02870a0d93840d952d905478637a3ea12b0dc8cc60c2f4362ad8bd51",
    "ViT_ReCo_swin_Validation_Checkpoint.zip": "1148dbcda3e3f9a16e5f780f4fd28ffb83b2ba618dcfd07e02235a88d4aa12e8",
}


def predictions(archive: ZipFile, stem: str):
    with np.load(BytesIO(archive.read(stem + ".npz"))) as data:
        return data["labels"].copy(), data["logits"].argmax(axis=1).copy()


for model in ("tiny", "small", "swin"):
    path = Path(__file__).resolve().parent / f"ViT_ReCo_{model}_Validation_Checkpoint.zip"
    with path.open("rb") as stream:
        actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    assert actual_hash == EXPECTED_SHA256[path.name], f"SHA-256 mismatch: {path.name}"
    with ZipFile(path) as archive:
        assert archive.testzip() is None, f"ZIP integrity failure: {path.name}"
        base = f"vitreco_validation/results/{model}/151/"
        labels, reco = predictions(archive, base + "vit_reco")
        qat_labels, qat = predictions(archive, base + "unpruned_qat")
        assert len(labels) == 3925 and np.array_equal(labels, qat_labels)
        a, b = reco == labels, qat == labels
        gained, lost = int(np.sum(a & ~b)), int(np.sum(~a & b))
        p = binomtest(gained, gained + lost, 0.5).pvalue
        reco_bytes = archive.getinfo(base + "vit_reco.vrc.zst").file_size
        qat_bytes = archive.getinfo(base + "unpruned_qat.vrc.zst").file_size
        recorded = {row["stage"]: row for row in json.loads(archive.read(base + "results.json"))}
        assert abs(100 * a.mean() - recorded["vit_reco"]["top1"]) < 1e-6
        assert abs(100 * b.mean() - recorded["unpruned_qat"]["top1"]) < 1e-6
        print(f"{model}: n={len(labels)}, ReCo={100*a.mean():.3f}%, QAT={100*b.mean():.3f}%, "
              f"difference={100*(a.mean()-b.mean()):+.3f} pp, gained/lost={gained}/{lost}, "
              f"McNemar p={p:.4f}, ReCo/QAT bytes={reco_bytes}/{qat_bytes}, "
              f"extra saving={100*(1-reco_bytes/qat_bytes):.3f}%")
