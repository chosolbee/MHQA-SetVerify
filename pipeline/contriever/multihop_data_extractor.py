import argparse
import json
import re
import sys
from collections import defaultdict
from tqdm import tqdm
from dataset import MultiHopDataset, get_dataset


def parse_chunks(dataset: MultiHopDataset):
    for sample in dataset:
        id = sample["id"]
        is_supports = sample["is_supports"]
        for idx, chunk in enumerate(sample["chunks"]):
            if is_supports[idx]:
                cid = f"{id}-sf-{idx:02d}"
            else:
                cid = f"{id}-{idx:02d}"
            yield {"id": cid, "text": chunk}


def purify_text(text: str):
    # delete all space and punctuations of the text
    pattern = r"[^\w]"
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text


def merge_chunks(chunks: list[dict]):
    chunk_mapping = defaultdict(set)
    pattern_title = r"<title>(.*?)</title>"

    for chunk in chunks:
        cid = chunk["id"]
        text = chunk["text"]
        key = purify_text(text)
        # chunk_mapping[text].add(cid)
        chunk_mapping[key].add((cid, text))

    chunks = []
    # for text, ids in chunk_mapping.items():
    for key, id_text_pairs in tqdm(chunk_mapping.items()):
        id_text_pairs = list(id_text_pairs)
        text = id_text_pairs[0][1]
        ids = [pair[0] for pair in id_text_pairs]
        ids = "//".join(list(ids))
        title = text.split(":")[0].strip()
        chunk_info = {"id": ids, "title": title, "text": text}
        chunks.append(chunk_info)
    return chunks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data Construction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-i", "--input", required=True, help="Input File Path", type=str)
    parser.add_argument("-o", "--output", required=True, help="Output File Path", type=str)
    
    return parser.parse_args()


def main(opt: argparse.Namespace):
    chunks = []
    dataset = get_dataset("musique", opt.input)
    for d in parse_chunks(dataset):
        chunks.append(d)
    chunks = merge_chunks(chunks)
    with open(opt.output, "w+") as f:
        for chunk in chunks:
            data = json.dumps(chunk)
            f.write(data + "\n")


if __name__ == "__main__":
    options = parse_args()
    main(options)
