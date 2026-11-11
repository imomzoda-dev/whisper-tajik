"""Read text/quality metadata (no audio) from every shard of Peacockery/tajik-asr-youtube."""
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq, pyarrow as pa
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
REPO = "datasets/Peacockery/tajik-asr-youtube/data"
shards = sorted(p.split("/")[-1] for p in fs.ls(REPO, detail=False))
COLS = ["text", "channel", "video_id", "clip_id", "duration", "scribe_wer", "scribe_cer"]
def read(name):
    with fs.open(f"{REPO}/{name}", block_size=2**20) as f:
        t = pq.ParquetFile(f).read(columns=COLS)
    t = t.append_column("shard", pa.array([name] * t.num_rows))
    t = t.append_column("row", pa.array(range(t.num_rows)))
    print("ok", name, t.num_rows, flush=True)
    return t
with ThreadPoolExecutor(6) as ex: tables = list(ex.map(read, shards))
pq.write_table(pa.concat_tables(tables), "data/youtube_meta.parquet")
print("DONE", flush=True)
