"""Build the training set.

  python scripts/prepare_data.py select --budget-gb 30   # choose best shards/clips from metadata
  python scripts/prepare_data.py fetch                   # download chosen shards, keep only good clips
  python scripts/prepare_data.py fleurs                  # FLEURS train/dev with original punctuation

Output: data/clips/<source>/<id>.flac  and  data/manifest_<split>.tsv  (path, text, duration, source)
"""
import argparse, collections, csv, glob, io, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tajik_text import fix_text, normalize_for_wer  # noqa: E402

YT_REPO = "Peacockery/tajik-asr-youtube"
V3 = "data/tajik-asr-corpus-v3/version=0"
SEL = "data/selection.json"
CLIPS = "data/clips"
lock = threading.Lock()


def v3_test_texts():
    s = set()
    for f in glob.glob(f"{V3}/corpus=*/split=test/**/*.parquet", recursive=True):
        s.update(normalize_for_wer(t) for t in pq.read_table(f, columns=["text"]).column("text").to_pylist())
    return s


def cyr_ratio(t):
    letters = [c for c in t if c.isalpha()]
    return sum("Ѐ" <= c <= "ӿ" for c in letters) / max(1, len(letters))


def select(a):
    import pandas as pd
    m = pd.read_parquet("data/youtube_meta.parquet")
    print(f"all clips: {len(m)}  hours: {m.duration.sum()/3600:.0f}")

    test = v3_test_texts()
    m["norm"] = m.text.fillna("").map(normalize_for_wer)
    bad_videos = set(m.loc[m.norm.isin(test) & (m.norm.str.len() > 15), "video_id"])
    print(f"test texts: {len(test)}  held-out videos excluded: {len(bad_videos)}")

    m["cps"] = m.norm.str.len() / m.duration.clip(lower=0.1)
    ok = (
        ~m.video_id.isin(bad_videos)
        & m.scribe_cer.notna() & (m.scribe_cer <= a.max_cer)
        & m.duration.between(1.0, 30.0)
        & m.cps.between(6, 25)
        & (m.norm.str.split().str.len() >= 2)
        & (m.text.fillna("").map(cyr_ratio) >= 0.97)
    )
    k = m[ok].copy()
    print(f"clean clips: {len(k)}  hours: {k.duration.sum()/3600:.0f}")
    print((k.groupby("channel").duration.sum() / 3600).round(1).sort_values(ascending=False).to_string())

    # Cap each channel so no single source (e.g. Quran readings) dominates.
    k = k.sample(frac=1, random_state=0)
    k["cum"] = k.groupby("channel").duration.cumsum() / 3600
    k = k[k.cum <= a.cap_hours]

    # Greedy: download the shards with the most useful hours per GB first.
    from huggingface_hub import HfApi
    sizes = {f.path.split("/")[-1]: f.size for f in HfApi().list_repo_tree(YT_REPO, "data", repo_type="dataset")}
    per = k.groupby("shard").duration.sum().sort_values(ascending=False) / 3600
    per = (per / per.index.map(lambda s: sizes[s] / 1e9)).sort_values(ascending=False)
    chosen, gb = [], 0.0
    for s in per.index:
        if gb + sizes[s] / 1e9 > a.budget_gb:
            continue
        chosen.append(s); gb += sizes[s] / 1e9
    k = k[k.shard.isin(chosen)]
    print(f"\nchosen shards: {len(chosen)}  download: {gb:.1f} GB  clean hours: {k.duration.sum()/3600:.0f}")
    print((k.groupby("channel").duration.sum() / 3600).round(1).sort_values(ascending=False).to_string())
    json.dump({"shards": {s: sorted(k[k.shard == s].row.astype(int).tolist()) for s in chosen},
               "excluded_videos": sorted(bad_videos)}, open(SEL, "w"))


def write_clip(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def fetch_one(shard, rows):
    from huggingface_hub import hf_hub_download
    done = f"data/done/{shard}.ok"
    if os.path.exists(done):
        return
    local = hf_hub_download(YT_REPO, f"data/{shard}", repo_type="dataset", local_dir="data/_yt_tmp")
    keep, lines, base = set(rows), [], 0
    # stream in small batches so a 2 GB shard never sits in RAM at once
    for b in pq.ParquetFile(local).iter_batches(batch_size=100, columns=["audio", "text", "clip_id", "duration", "channel"]):
        for j, r in enumerate(b.to_pylist()):
            if base + j not in keep:
                continue
            p = f"{CLIPS}/yt/{r['channel']}/{r['clip_id']}.flac".replace(" ", "_")
            write_clip(p, r["audio"]["bytes"])
            lines.append((p, fix_text(r["text"]), r["duration"], "yt-" + r["channel"]))
        base += b.num_rows
    with lock:
        with open("data/manifest_yt.tsv", "a", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(lines)
    os.remove(local)
    os.makedirs("data/done", exist_ok=True)
    open(done, "w").close()
    print(f"ok {shard} kept {len(lines)}", flush=True)


def fetch(a):
    sel = json.load(open(SEL))["shards"]
    todo = [(s, r) for s, r in sel.items() if not os.path.exists(f"data/done/{s}.ok")]
    print(f"shards to fetch: {len(todo)} / {len(sel)}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for f in [ex.submit(fetch_one, s, r) for s, r in todo]:
            f.result()
    print("DONE", flush=True)


def fleurs(a):
    """FLEURS audio from the v3 corpus + original cased/punctuated text from google/fleurs."""
    import urllib.request
    from huggingface_hub import hf_hub_download
    for split, out in [("train", "train"), ("dev", "dev")]:
        p = f"version=0/corpus=fleurs/split={split}/language=tgk_Cyrl/part-00000.parquet"
        hf_hub_download("Peacockery/tajik-asr-corpus-v3", p, repo_type="dataset", local_dir="data/tajik-asr-corpus-v3")
        gsplit = {"train": "train", "dev": "dev"}[split]
        url = f"https://huggingface.co/datasets/google/fleurs/resolve/main/data/tg_tj/{gsplit}.tsv"
        raw = {}
        for line in urllib.request.urlopen(url).read().decode().splitlines():
            c = line.split("\t")
            raw[normalize_for_wer(c[2])] = c[2]
        rows = pq.read_table(f"data/tajik-asr-corpus-v3/{p}").to_pylist()
        lines, miss = [], 0
        for i, r in enumerate(rows):
            text = raw.get(normalize_for_wer(r["text"]))
            if text is None:
                miss += 1; continue
            b = np.asarray(r["audio_bytes"], dtype=np.int8).view(np.uint8).tobytes()
            path = f"{CLIPS}/fleurs/{split}/{i:05d}.flac"
            write_clip(path, b)
            lines.append((path, fix_text(text), r["audio_size"] / 16000, "fleurs"))
        with open(f"data/manifest_fleurs_{out}.tsv", "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(lines)
        print(f"fleurs {split}: {len(lines)} clips, unmatched {miss}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "fetch", "fleurs"])
    ap.add_argument("--budget-gb", type=float, default=30)
    ap.add_argument("--max-cer", type=float, default=0.04)
    ap.add_argument("--cap-hours", type=float, default=25)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"select": select, "fetch": fetch, "fleurs": fleurs}[a.cmd](a)
