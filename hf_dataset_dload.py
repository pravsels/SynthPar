import os
import time
from huggingface_hub import hf_hub_download, list_repo_files
from zipfile import ZipFile
from tqdm import tqdm
import requests

dataset_name = "synthpar2"
hf_repo_id = f"pravsels/{dataset_name}"
dataset_path = "./"
dataset_dir = os.path.join(dataset_path, dataset_name)
os.makedirs(dataset_dir, exist_ok=True)

def download_and_extract_shard(shard_filename, max_retries=50, retry_delay=10):
    for attempt in range(max_retries):
        try:
            shard_path = hf_hub_download(repo_id=hf_repo_id,
                                         repo_type="dataset",
                                         filename=shard_filename,
                                         local_dir=dataset_path)
            
            with ZipFile(shard_path, "r") as zip_ref:
                total_files = len(zip_ref.namelist())
                progress_bar = tqdm(total=total_files,
                                    unit="file",
                                    desc=f"Extracting images from {shard_filename}")
                
                sorted_filenames = sorted(zip_ref.namelist())
                for file in sorted_filenames:
                    zip_ref.extract(file, dataset_dir)
                    progress_bar.update(1)
                
                progress_bar.close()
            
            os.remove(shard_path)
            return True
        except (requests.exceptions.RequestException, OSError) as e:
            print(f"Error downloading or extracting {shard_filename}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached for {shard_filename}. Skipping.")
                return False

# Get the list of files in the repository
try:
    repo_files = list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
except Exception as e:
    print(f"Error listing repository files: {e}")
    repo_files = []

# Filter for shard files
shard_files = [f for f in repo_files if f.startswith("shard_") and f.endswith(".zip")]

st_type = input("Enter ST type (e.g. ST1, ST2): ")
if not st_type.startswith("ST") or not st_type[2:].isdigit() or not 1 <= int(st_type[2:]) <= 8:
    print("Invalid input. Please enter ST1 through ST8")
    exit() 

filter_st = lambda files, st: list(filter(lambda x: st in x, files))
filtered_files = sorted(filter_st(shard_files, st_type), key=lambda x: int(x.split('_')[1]))

print('first 5 in the list of filtered files : ', filtered_files[:5])

for shard_filename in filtered_files:
    success = download_and_extract_shard(shard_filename)
    if not success:
        print(f"Failed to process {shard_filename}")

print("Download and extraction process completed.")

