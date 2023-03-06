from tqdm.auto import tqdm
import requests
import re
import shutil
import os

def download_model(url:str):
    """ Downloads the model's weigths in a zip file and extracts it in the cache
    Arg:
        url (str): URL of the model
    
    Returns:
        model_path (srt): the file path of the model in the cache
    """

    PROJECT_ROOT = os.getenv("ROOT")
    if PROJECT_ROOT is None:
        raise Exception("Environment variable ROOT does not exist. Please add it in the file env/bin/activate and reactivate the environnement.")

    if not(os.path.exists(os.path.join(PROJECT_ROOT, ".cache"))):
        os.makedirs(os.path.join(PROJECT_ROOT, ".cache"))
    
    CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")

    res = requests.get(url, stream = True)
    filename = re.findall(r'filename="([a-zA-Z0-9_\-.]*)"', res.headers['Content-Disposition'])[0]
    content_length = int(res.headers['Content-Length'])
    chunk_size = 8192

    basename = filename.split('.')[0]

    filename = os.path.join(CACHE_DIR, filename)
    folder = os.path.join(CACHE_DIR, basename)

    if os.path.exists(folder):
        print(f"File {filename} has already been downloaded, skipping download")
    else:
        print(f"Downloading {filename} ...")
        with open(filename, "wb") as f:
            for chunk in tqdm(iterable=res.iter_content(chunk_size=chunk_size), total=content_length/chunk_size, unit='KB'):
                f.write(chunk)
                
        print("Done.\n")
        print("Extracting files ...")
        shutil.unpack_archive(filename, extract_dir=CACHE_DIR)
        if os.path.exists(os.path.join(CACHE_DIR, "__MACOSX")):
            # For macOS users
            shutil.rmtree(os.path.join(CACHE_DIR, "__MACOSX"))
        print("Done.")
        print(f"Deleting {filename} ...")
        os.remove(filename)
        print("Done.")

    model_path = f"{folder}/{basename}_weights"

    return model_path