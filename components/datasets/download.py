from tqdm import tqdm
import requests
import shutil
from pathlib import Path
import math

def download_file(path, url):
    response = requests.get(url, stream=True)
    total_length = math.ceil(float(response.headers.get('content-length'))/4096)

    with open(path, "wb") as handle:
        for data in tqdm(response.iter_content(chunk_size=4096), total=total_length):
            handle.write(data)

def unpack_data(path):
    path = Path(path).absolute()
    shutil.unpack_archive(path, path.parent)
