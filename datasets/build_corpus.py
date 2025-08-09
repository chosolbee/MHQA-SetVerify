import argparse
import json
import tarfile
import bz2
from tqdm import tqdm
from config import DATASET_PATHS


def build_hotpotqa_corpus(input_path):
    corpus_path = DATASET_PATHS["hotpotqa"]["passages"]

    with tarfile.open(input_path, mode="r|bz2") as tar, \
         open(corpus_path, "w", encoding="utf-8") as out:
        for member in tar:
            print(f"Processing {member.name}...")

            if not member.isfile() or not member.name.endswith(".bz2"):
                continue

            comp_file = tar.extractfile(member)
            if comp_file is None:
                continue

            decompressor = bz2.BZ2Decompressor()
            buffer = ""

            for chunk in iter(lambda: comp_file.read(64*1024), b""):
                data = decompressor.decompress(chunk).decode("utf-8")
                buffer += data

                lines = buffer.split("\n")
                buffer = lines.pop()

                for line in lines:
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj.get("text"), list):
                        obj["text"] = "".join(obj["text"])
                    out.write(json.dumps(obj, ensure_ascii=False))
                    out.write("\n")

            if buffer.strip():
                obj = json.loads(buffer)
                if isinstance(obj.get("text"), list):
                    obj["text"] = "".join(obj["text"])
                out.write(json.dumps(obj, ensure_ascii=False))
                out.write("\n")

    print("Generated corpus for HotpotQA in", corpus_path)


def build_2wikimultihopqa_corpus():
    dataset_paths = DATASET_PATHS["2wikimultihopqa"]
    corpus = {}

    for dataset_type in ["train", "dev", "test"]:
        dataset_path = dataset_paths[dataset_type]
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for question in tqdm(dataset, desc=f"Processing {dataset_path}"):
            question_id = question["_id"]
            context = question["context"]
            supporting_facts = [sf[0] for sf in question["supporting_facts"]]

            for idx, doc in enumerate(context):
                title = doc[0]
                text = " ".join(doc[1])
                is_supporting = title in supporting_facts

                hashed_doc = hash(f"{title}: {text}")
                new_id = f"{question_id}{'-sf' if is_supporting else ''}-{idx:02d}"

                if hashed_doc not in corpus:
                    corpus[hashed_doc] = {
                        "id": new_id,
                        "title": title,
                        "text": text,
                    }
                else:
                    existing_doc = corpus[hashed_doc]
                    existing_doc["id"] += f"//{new_id}"

    print(f"Generated corpus with {len(corpus)} unique documents.")

    with open(dataset_paths["passages"], "w", encoding="utf-8") as f:
        for doc in corpus.values():
            f.write(json.dumps(doc) + "\n")

    print("Generated corpus for 2WikiMultiHopQA in", dataset_paths["passages"])


def build_musique_corpus():
    dataset_paths = DATASET_PATHS["musique"]
    corpus = {}

    for dataset_type in ["train", "dev", "test"]:
        dataset_path = dataset_paths[dataset_type]
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line.strip()) for line in f]

        for question in tqdm(dataset, desc=f"Processing {dataset_path}"):
            question_id = question["id"]
            context = question["paragraphs"]

            for doc in context:
                idx = doc["idx"]
                title = doc["title"]
                text = doc["paragraph_text"]
                is_supporting = doc.get("is_supporting", False)

                hashed_doc = hash(f"{title}: {text}")
                new_id = f"{question_id}{'-sf' if is_supporting else ''}-{idx:02d}"

                if hashed_doc not in corpus:
                    corpus[hashed_doc] = {
                        "id": new_id,
                        "title": title,
                        "text": text,
                    }
                else:
                    existing_doc = corpus[hashed_doc]
                    existing_doc["id"] += f"//{new_id}"

    print(f"Generated corpus with {len(corpus)} unique documents.")

    with open(dataset_paths["passages"], "w", encoding="utf-8") as f:
        for doc in corpus.values():
            f.write(json.dumps(doc) + "\n")

    print("Generated corpus for MuSiQue in", dataset_paths["passages"])


def parse_args():
    parser = argparse.ArgumentParser(description="Make Corpus from Dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=["hotpotqa", "2wikimultihopqa", "musique"], help="Dataset name")
    parser.add_argument("--input-path", type=str, help="Input path for the dataset (if needed)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "hotpotqa":
        build_hotpotqa_corpus(args.input_path)
    elif args.dataset == "2wikimultihopqa":
        build_2wikimultihopqa_corpus()
    elif args.dataset == "musique":
        build_musique_corpus()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
