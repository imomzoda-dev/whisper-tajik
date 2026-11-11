"""Download Peacockery/tajik-asr-corpus-v3 parquet shards.
phase 1: FLEURS + every test split + train shards smaller than MAX_GB.
phase 2: everything."""
import sys
from huggingface_hub import HfApi, hf_hub_download

REPO = "Peacockery/tajik-asr-corpus-v3"
OUT = "data/tajik-asr-corpus-v3"
phase = sys.argv[1] if len(sys.argv) > 1 else "1"
MAX_BYTES = 3e9

api = HfApi()
files = [f for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)
         if getattr(f, "size", None) is not None and f.path.endswith(".parquet")]
want = []
for f in files:
    if phase == "2" or "corpus=fleurs" in f.path or "split=test" in f.path or "split=dev" in f.path or f.size < MAX_BYTES:
        want.append(f)
want.sort(key=lambda f: f.size)
print(f"phase {phase}: {len(want)} files, {sum(f.size for f in want)/1e9:.1f} GB", flush=True)
for i, f in enumerate(want, 1):
    hf_hub_download(REPO, f.path, repo_type="dataset", local_dir=OUT)
    print(f"[{i}/{len(want)}] {f.size/1e6:.0f} MB {f.path}", flush=True)
print("DONE", flush=True)
