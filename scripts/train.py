"""Continue fine-tuning a Tajik Whisper model on Apple Silicon (MPS) or CUDA.

usage: python scripts/train.py --model models/whisper-small-tajik-base --out models/run1 \
           --train data/manifest_yt.tsv data/manifest_fleurs_train.tsv --dev data/manifest_fleurs_dev.tsv
"""
import argparse, csv, math, os, random, signal, sys, time

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tajik_text import fix_text  # noqa: E402


def read_manifest(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows += [(r[0], r[1], float(r[2]), r[3]) for r in csv.reader(f, delimiter="\t")]
    return rows


class Clips(Dataset):
    def __init__(self, rows, proc, augment=False):
        self.rows, self.proc, self.augment = rows, proc, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, text, _, _ = self.rows[i]
        a, sr = sf.read(path, dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        if self.augment:
            a = a * random.uniform(0.5, 1.5)                     # loudness
            if random.random() < 0.3:                            # light background noise
                a = a + np.random.randn(len(a)).astype("float32") * random.uniform(0.001, 0.01)
            if random.random() < 0.3:                            # speed perturbation 0.9-1.1x
                r = random.uniform(0.9, 1.1)
                idx = np.arange(0, len(a), r)
                a = np.interp(idx, np.arange(len(a)), a).astype("float32")
            a = np.clip(a, -1, 1)
        a = a[: 30 * 16000]
        feats = self.proc.feature_extractor(a, sampling_rate=16000).input_features[0]
        labels = self.proc.tokenizer(fix_text(text)).input_ids
        return feats, labels


class Collate:  # a class, not a closure: macOS DataLoader workers must pickle it
    def __init__(self, sot):
        self.sot = sot

    def __call__(self, batch):
        feats = torch.tensor(np.stack([b[0] for b in batch]))
        labs = [b[1][1:] if b[1][0] == self.sot else b[1] for b in batch]   # model prepends SOT itself
        labs = [l[:448] for l in labs]
        # round up to a multiple of 64: few distinct shapes keep the MPS allocator cache from ballooning
        n = min(448, -(-max(len(l) for l in labs) // 64) * 64)
        labels = torch.full((len(labs), n), -100, dtype=torch.long)
        for i, l in enumerate(labs):
            labels[i, : len(l)] = torch.tensor(l)
        return feats, labels


@torch.no_grad()
def dev_loss(model, dl, device, dtype, max_batches=40):
    model.eval()
    tot, n = 0.0, 0
    for i, (x, y) in enumerate(dl):
        if i >= max_batches:
            break
        with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
            tot += model(input_features=x.to(device), labels=y.to(device)).loss.item(); n += 1
    model.train()
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    torch.manual_seed(0); random.seed(0)

    proc = WhisperProcessor.from_pretrained(a.model)
    proc.tokenizer.set_prefix_tokens(language="tajik", task="transcribe", predict_timestamps=False)
    src = a.out + "/last" if a.resume and os.path.isdir(a.out + "/last") else a.model
    # Dropout must stay 0: this model was trained without it, and turning it on makes the train-mode
    # loss jump from ~0.3 to ~7 (measured). Dropout is read when layers are built, so it is passed to
    # from_pretrained; setting config.dropout afterwards would silently do nothing.
    model = WhisperForConditionalGeneration.from_pretrained(
        src, dropout=0.0, attention_dropout=0.0, activation_dropout=0.0).to(device)
    # Regularise with SpecAugment instead (read at run time, so setting the config here works).
    model.config.apply_spec_augment = True
    model.config.mask_time_prob, model.config.mask_feature_prob, model.config.mask_feature_length = 0.05, 0.05, 10
    assert model.model.encoder.layers[0].dropout == 0.0
    model.generation_config.language, model.generation_config.task = "tajik", "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()

    train_rows = read_manifest(a.train)
    dev_rows = read_manifest(a.dev)
    print(f"train {len(train_rows)} clips / {sum(r[2] for r in train_rows)/3600:.1f} h, dev {len(dev_rows)}", flush=True)

    collate = Collate(proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"))
    dl = DataLoader(Clips(train_rows, proc, augment=True), batch_size=a.bs, shuffle=True, num_workers=a.workers,
                    collate_fn=collate, drop_last=True, persistent_workers=True, prefetch_factor=4)
    dev_dl = DataLoader(Clips(dev_rows, proc), batch_size=a.bs, shuffle=False, num_workers=2, collate_fn=collate)

    total = math.ceil(len(dl) * a.epochs / a.accum)
    if a.max_steps:
        total = min(total, a.max_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, a.warmup, total)
    step, best = 0, None
    state_path = a.out + "/last/train_state.pt"
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, weights_only=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); step = st["step"]
        best = st.get("best")
        print(f"resumed at step {step}", flush=True)

    def save(tag):
        d = f"{a.out}/{tag}"
        model.save_pretrained(d); proc.save_pretrained(d)
        if tag == "last":
            torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "best": best}, state_path)

    # graceful stop: SIGTERM (sent by ./tajik.sh stop) -> save progress after the current step, then exit
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    os.makedirs(a.out, exist_ok=True)
    if best is None:
        best = dev_loss(model, dev_dl, device, dtype)
        print(f"step {step} dev_loss {best:.4f}", flush=True)
    else:
        print(f"best dev_loss so far {best:.4f}", flush=True)
    t0, run_loss, micro = time.time(), 0.0, 0
    while step < total:
        for x, y in dl:
            with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                loss = model(input_features=x.to(device), labels=y.to(device)).loss / a.accum
            loss.backward()
            run_loss += loss.item(); micro += 1
            if micro % a.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if device == "mps" and step % 10 == 0:
                torch.mps.empty_cache()
            if step % a.log_every == 0:
                el = time.time() - t0
                mem = f" mem {torch.mps.driver_allocated_memory()/1e9:.1f}GB" if device == "mps" else ""
                print(f"step {step}/{total} loss {run_loss/a.log_every:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{el/a.log_every:.2f}s/step eta {(total-step)*el/a.log_every/3600:.1f}h{mem}", flush=True)
                run_loss, t0 = 0.0, time.time()
            if step % a.save_every == 0 or step == total:
                dl_ = dev_loss(model, dev_dl, device, dtype)
                print(f"step {step} dev_loss {dl_:.4f}", flush=True)
                save("last")
                if dl_ < best:
                    best = dl_; save("best"); print("  new best", flush=True)
            if stop["flag"]:
                save("last")
                print(f"STOPPED at step {step}/{total}, progress saved", flush=True)
                sys.exit(3)
            if step >= total:
                break
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
