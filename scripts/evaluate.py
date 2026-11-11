"""Evaluate a Whisper checkpoint on Tajik test sets.

usage: python scripts/evaluate.py MODEL_DIR [--set fleurs|youtube|all] [--limit N] [--beams 1]
Prints WER/CER (lowercase, no punctuation) and writes predictions to results/.
"""
import argparse, glob, io, json, os, sys, time

import jiwer
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tajik_text import normalize_for_wer  # noqa: E402

ROOT = "data/tajik-asr-corpus-v3/version=0"
SETS = {
    "fleurs": [f"{ROOT}/corpus=fleurs/split=test"],
    "youtube": sorted(p for p in glob.glob(f"{ROOT}/corpus=youtube-*/split=test")),
}


def load_v3(dirs, limit=None):
    items = []
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/**/*.parquet", recursive=True)):
            for r in pq.read_table(f).to_pylist():
                b = np.asarray(r["audio_bytes"], dtype=np.int8).view(np.uint8).tobytes()
                a, sr = sf.read(io.BytesIO(b), dtype="float32")
                assert sr == 16000
                items.append((a, r["text"]))
    if limit:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(items))[:limit]
        items = [items[i] for i in sorted(idx)]
    return items


@torch.inference_mode()
def transcribe(model, proc, audios, device, bs=16, beams=1):
    out = []
    for i in range(0, len(audios), bs):
        feats = proc.feature_extractor(audios[i:i + bs], sampling_rate=16000, return_tensors="pt").input_features
        ids = model.generate(feats.to(device, model.dtype), language="tg", task="transcribe",
                             num_beams=beams, max_new_tokens=220)
        out += proc.batch_decode(ids, skip_special_tokens=True)
        print(f"  {min(i + bs, len(audios))}/{len(audios)}", end="\r", flush=True)
    return [o.strip() for o in out]


def score(refs, hyps):
    r = [normalize_for_wer(x) or "<empty>" for x in refs]
    h = [normalize_for_wer(x) for x in hyps]
    return 100 * jiwer.wer(r, h), 100 * jiwer.cer(r, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--set", default="fleurs", choices=["fleurs", "youtube", "all"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = WhisperProcessor.from_pretrained(a.model)
    model = WhisperForConditionalGeneration.from_pretrained(a.model, dtype=torch.float16).to(device).eval()

    os.makedirs("results", exist_ok=True)
    names = ["fleurs", "youtube"] if a.set == "all" else [a.set]
    summary = {}
    for name in names:
        data = load_v3(SETS[name], a.limit)
        t0 = time.time()
        hyps = transcribe(model, proc, [x[0] for x in data], device, a.bs, a.beams)
        refs = [x[1] for x in data]
        wer, cer = score(refs, hyps)
        summary[name] = {"n": len(data), "wer": round(wer, 2), "cer": round(cer, 2),
                         "sec": round(time.time() - t0)}
        print(f"\n{name}: n={len(data)}  WER={wer:.2f}%  CER={cer:.2f}%")
        tag = os.path.basename(os.path.normpath(a.model))
        with open(f"results/{tag}_{name}_b{a.beams}.jsonl", "w") as f:
            for r, h in zip(refs, hyps):
                f.write(json.dumps({"ref": r, "hyp": h}, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
