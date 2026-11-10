import urllib.request
import os
import time

dest_dir = "D:/models/distilbert"
os.makedirs(dest_dir, exist_ok=True)

files = [
    "config.json",
    "tokenizer_config.json",
    "vocab.txt",
    "model.safetensors"
]
base_url = "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/"

def reporthook(count, block_size, total_size):
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        mb = count * block_size / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        if count % 500 == 0 or percent >= 100:
            print(f"Downloaded {mb:.1f} MB / {total_mb:.1f} MB ({percent}%)", flush=True)

for f in files:
    target_path = os.path.join(dest_dir, f)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        print(f"Already exists: {f} ({os.path.getsize(target_path)} bytes)")
        continue
    url = base_url + f
    print(f"Downloading {f} from {url}...")
    urllib.request.urlretrieve(url, target_path, reporthook=reporthook)
    print(f"Finished {f}: {os.path.getsize(target_path)} bytes")

print("All files downloaded to", dest_dir)
