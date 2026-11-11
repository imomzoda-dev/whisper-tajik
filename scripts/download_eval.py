"""Download evaluation sets from Peacockery/tajik-asr-corpus-v3: every test/dev split (FLEURS + held-out YouTube videos)."""
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download
REPO = "Peacockery/tajik-asr-corpus-v3"
files = [f.path for f in HfApi().list_repo_tree(REPO, repo_type="dataset", recursive=True)
         if f.path.endswith(".parquet") and ("split=test" in f.path or "split=dev" in f.path)]
def get(p):
    hf_hub_download(REPO, p, repo_type="dataset", local_dir="data/tajik-asr-corpus-v3"); print("ok", p, flush=True)
with ThreadPoolExecutor(3) as ex: list(ex.map(get, files))
print("DONE", len(files), flush=True)
