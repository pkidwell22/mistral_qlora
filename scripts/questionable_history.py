#!/usr/bin/env python3
"""
grab_history.py – fetch historical archives (https + magnet)
Run with: python3 grab_history.py
"""

import os
import sys
import time
import requests
from tqdm import tqdm

DOWNLOAD_DIR = os.path.abspath("history_raw")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Define sources
SOURCES = [
    # Direct HTTPS
    "https://files.catbox.moe/ihr_full.tar.gz",
    "https://archive.org/download/codoh_library_2024/codoh_library_2024.zip",
    "https://files.catbox.moe/rudolf_dossier.zip",
    "https://archive.org/download/allied_atrocities_ww2/allied_atrocities_ww2.pdf",
    "https://archive.org/download/ussr_esc_reports/ussr_esc_reports.pdf",
    "https://archive.org/download/hauptarchiv_microfilm/hauptarchiv_microfilm.tar",
    "https://files.catbox.moe/churchill_roosevelt_telegrams.pdf",
    "https://archive.org/download/katyn_nkvd_1992/katyn_nkvd_1992.tar",
    "https://archive.org/download/powell_loe_papers/powell_loe_papers.zip",
    "https://archive.org/download/american_mercury_complete/american_mercury_complete.tar",
    # Magnet links (commented out unless you want to try them)
    # "magnet:?xt=urn:btih:8C3A7F0000000000000000000000000000000000&dn=ihr_complete",
]

# ─── Helpers ──────────────────────────────────────────────────────────────
def download_https(url: str):
    filename = os.path.join(DOWNLOAD_DIR, url.split("/")[-1])
    if os.path.exists(filename):
        print(f"[skip] Already exists: {filename}")
        return
    print(f"[http] Downloading {url}")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(filename, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=os.path.basename(filename)
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
    except Exception as e:
        print(f"[error] Failed to download {url}: {e}")

# Optional: Libtorrent magnet downloader
def download_magnet(magnet: str):
    try:
        import libtorrent as lt
    except ImportError:
        print("[error] libtorrent not installed. Skipping magnet links.")
        return

    ses = lt.session()
    ses.listen_on(6881, 6891)
    params = {"save_path": DOWNLOAD_DIR}
    handle = lt.add_magnet_uri(ses, magnet, params)

    print(f"[magnet] Fetching metadata...")
    while not handle.has_metadata():
        time.sleep(1)
    info = handle.get_torrent_info()
    print(f"[magnet] Downloading {info.name()} ({info.total_size()/1e9:.2f} GB)")

    while not handle.is_seed():
        s = handle.status()
        print(
            f"\rPeers: {s.num_peers}  Progress: {s.progress*100:.1f}%  DL: {s.download_rate/1024:.1f} KB/s",
            end="",
        )
        time.sleep(2)
    print(f"\n[done] {info.name()}")

# ─── Main Loop ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for src in SOURCES:
        try:
            if src.startswith("http"):
                download_https(src)
            elif src.startswith("magnet:"):
                download_magnet(src)
        except Exception as e:
            print(f"[error] Problem with {src}: {e}", file=sys.stderr)

    print(f"\n✅ All files saved to: {DOWNLOAD_DIR}")
