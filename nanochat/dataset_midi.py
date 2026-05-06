"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq

from multiprocessing import Pool
# from midi_utils import *
from huggingface_hub import HfFileSystem
from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/drengskapur/midi-classical-music/resolve/main/data"
BASE_URL = "https://huggingface.co/datasets/SyMuPe/PianoCoRe/resolve/main/data"
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet
# index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_midi")

fs = HfFileSystem()
fname_list = fs.ls('datasets/drengskapur/midi-classical-music/data')
fname_list = fs.ls('datasets/SyMuPe/PianoCoRe/data')

MAX_SHARD = min(MAX_SHARD, len(fname_list)-1)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

# def get_midi_string(split, start=0, step=1):
#     assert split in ["train", "val"], "split must be 'train' or 'val'"

#     fname_list = os.listdir(DATA_DIR)
#     for fname in fname_list:
#         midi_path_dir = 'tmp_data'
#         midi_path = 'albeniz-aragon_fantasia_op47_part_6.mid'
#         midi_path = os.path.join(midi_path_dir,fname)

#         # Load the MIDI file
#         imported_midi = import_midi(midi_path,midi_processor_highest_pitch_track)
#         imported_midi = imported_midi.astype(float)
#         imported_midi = imported_midi[:,:3]

#         midi_notes = imported_midi

#         # sort by time 
#         # sorted_inds = np.argsort(midi_notes[:,0])
#         sorted_inds = np.lexsort((midi_notes[:,0],midi_notes[:,1],midi_notes[:,2]))
#         midi_notes_sorted = midi_notes[sorted_inds,:]

#         highest_notes = midi_notes_sorted

#         # get offset differences
#         highest_notes[1:,0] = highest_notes[1:,0] - highest_notes[:-1,0]
#         highest_notes[0,0] = 0

#         sequence = 'BOS_None '
#         for val in imported_midi:
#             sequence += '%1.3f'%(val[0]) + '-' + '%1.3f'%(val[1]) + '-' + str(int(val[2])) + ' '
#         # sequence += ' [EOS]'

#         yield sequence

# -----------------------------------------------------------------------------

def download_single_file(index):
    """ Downloads a single file index, with some backoff """
    
    # Construct the local filepath for this file and skip if it already exists
    # print(fname_list[index]['name'])
    filename = fname_list[index]['name'].split('/')[-1]
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of train shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    # Prepare the output directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # The way this works is that the user specifies the number of train shards to download via the -n flag.
    # In addition to that, the validation shard is *always* downloaded and is pinned to be the last shard.
    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD) # always download the validation shard

    # Download the shards
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()

    # fname_list = fs.ls('datasets/drengskapur/midi-classical-music/data')
    # print

    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
