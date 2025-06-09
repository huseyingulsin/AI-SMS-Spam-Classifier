# dataset/download.py

import requests
import zipfile
import io
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
EXTRACT_TO = Path("dataset")

def download_dataset():
    print("[*] Downloading dataset...")
    response = requests.get(URL)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(EXTRACT_TO)

    print("[+] Dataset extracted to", EXTRACT_TO)

if __name__ == "__main__":
    download_dataset()
