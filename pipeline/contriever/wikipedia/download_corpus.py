import os
import gzip
import requests
import shutil

def download_and_extract_gzip(url, output_dir):
    print("Download and Extract")
    os.makedirs(output_dir, exist_ok=True)
    local_gz_path = os.path.join(output_dir, os.path.basename(url))
    print("Downloading corpus ...")
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise Exception("Download failed: " + str(r.status_code))
    
    with open(local_gz_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    print("Extracting gzip file ...")
    output_file = local_gz_path[:-3]
    with gzip.open(local_gz_path, 'rb') as f_in, open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    print("Done.")

if __name__ == "__main__":
    corpus_url = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"  
    output_directory = ""
    
    print("output_directory")

    download_and_extract_gzip(corpus_url, output_directory)
