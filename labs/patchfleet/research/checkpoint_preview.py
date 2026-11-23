"""Measure exact early-start corrections in the first MLP of a trained Pythia checkpoint."""

import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen
import numpy as np
from safetensors import safe_open
from scipy.special import erf
from tokenizers import Tokenizer
from threadpoolctl import threadpool_limits

REV = "a39f36b100fe8a5377810d56c3f4789b9c53ac42"
PINS = {
    "model.safetensors": "ebfa4e2f18696ebd83716a0d39fe2c025f2ff8483f72a83ca59c475692fc9d15",
    "tokenizer.json": "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624",
    "config.json": "002050231a9b1ec3ac77aa6b9b3bbdc4d923f4068a7dd33b8da72a9bd6ad9a43",
}


def measure(cache):
    cache.mkdir(parents=True, exist_ok=True)
    for name, digest in PINS.items():
        path = cache / name
        if not path.exists():
            with urlopen(
                f"https://huggingface.co/EleutherAI/pythia-70m/resolve/{REV}/{name}?download=true",
                timeout=60,
            ) as res:
                path.write_bytes(res.read())
        with path.open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == digest
    cfg = json.loads((cache / "config.json").read_text())
    assert cfg["use_parallel_residual"] and cfg["hidden_act"] == "gelu"
    tok = Tokenizer.from_file(str(cache / "tokenizer.json"))
    samples = json.loads(
        (Path(__file__).resolve().parents[1] / "site/cases.json").read_text()
    )
    tokens = []
    for sample in samples:
        with urlopen(sample["source"], timeout=30) as res:
            data = res.read()
        assert (
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            == sample["before"]
        )
        tokens.extend(tok.encode(data.decode(), add_special_tokens=False).ids[:512])
    with safe_open(str(cache / "model.safetensors"), framework="numpy") as model:

        def weight(name):
            return model.get_tensor("gpt_neox.layers.0." + name).astype("f")

        x = model.get_tensor("gpt_neox.embed_in.weight")[tokens].astype("f")
        scale = weight("post_attention_layernorm.weight")
        bias = weight("post_attention_layernorm.bias")
        x = (x - x.mean(axis=1, keepdims=True)) / np.sqrt(
            x.var(axis=1, keepdims=True) + cfg["layer_norm_eps"]
        ) * scale + bias
        up = weight("mlp.dense_h_to_4h.weight").T
        up_bias = weight("mlp.dense_h_to_4h.bias")
        down = weight("mlp.dense_4h_to_h.weight").T

    def activate(z):
        return 0.5 * z * (1 + erf(z / np.sqrt(np.float32(2))))

    rows = []
    for missing in (1, 8, 64):
        # Partition the actual predecessor projection. Both schedules use the
        # same reduction order to isolate the late contribution from FP reassociation.
        preview = x[:, :-missing] @ up[:-missing] + up_bias
        remainder = x[:, -missing:] @ up[-missing:]
        h = activate(preview + remainder)
        guess = activate(preview)
        delta = h - guess
        y = h @ down
        y_guess = guess @ down
        error = float(np.linalg.norm(y - (y_guess + delta @ down)) / np.linalg.norm(y))
        assert error < 1e-4
        rows.append(
            dict(
                late_input_channels=missing,
                preview_channels=x.shape[1] - missing,
                corrected_fraction_fp32=float(np.mean(delta != 0)),
                corrected_fraction_fp16=float(
                    np.mean(h.astype("f2") != guess.astype("f2"))
                ),
                relative_error_using_preview_alone=float(
                    np.linalg.norm(y - y_guess) / np.linalg.norm(y)
                ),
                relative_error_after_correction=error,
            )
        )
    return dict(
        model="EleutherAI/pythia-70m",
        revision=REV,
        tokens=len(tokens),
        hidden_width=x.shape[1],
        mlp_width=up.shape[1],
        dtype="float16 checkpoint, fp32 matrix products and GELU",
        results=rows,
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        Path(sys.argv[2]).write_text(
            json.dumps(measure(Path(sys.argv[1])), indent=2) + "\n"
        )
