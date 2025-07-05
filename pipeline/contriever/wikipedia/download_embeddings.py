import os
import tarfile
import requests

def download_and_extract_tar(url, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    local_tar_path = os.path.join(output_dir, os.path.basename(url))
    print("Downloading pre-computed embeddings ...", flush=True)
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise Exception("Download failed: " + str(r.status_code))
    
    with open(local_tar_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    print("Extracting tar file ...", flush=True)
    with tarfile.open(local_tar_path, "r") as tar:
        tar.extractall(path=output_dir)
    print("Done.", flush=True)

if __name__ == "__main__":
    # Contriever 버전
    url = "https://dl.fbaipublicfiles.com/contriever/embeddings/contriever/wikipedia_embeddings.tar"
    
    # Contriever-msmarco 버전
    # url = "https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar"
    
    output_directory = ""
    download_and_extract_tar(url, output_directory)
