import os
import sys
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, set_seed
from ..multihead_stop_decider_train import MultiheadModel
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from pipeline.answer_generator.prompts import gen_final_answer_prompt, gen_final_answer_docs_only_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Compute answer scores from the dataset")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for processing")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model ID for computing scores")
    parser.add_argument("--max-length", type=int, default=4096, help="Max Length of Inputs")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision for model")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the checkpoint of the multihead classifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use-docs-only", action="store_true", help="Use only documents from trace")

    return parser.parse_args()

def extract_documents_only(trace_text):
    documents = []
    lines = trace_text.split('\n')
    for line in lines:
        if line.startswith("Document: "):
            documents.append(line)
    return '\n'.join(documents)

if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)

    model = MultiheadModel.from_pretrained(
        args.checkpoint_path,
        encoder_kwargs={
            "device_map": "auto",
            "use_cache": False,
            "max_position_embeddings": args.max_length,
        },
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        inference_mode=True,
    )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print("Model and tokenizer loaded successfully.", flush=True)

    with open(args.input_path, "r", encoding="utf-8") as f:
        traces = [json.loads(line.strip()) for line in f]

    open(args.output_path, "w", encoding="utf-8").close()

    for i in tqdm(range(0, len(traces), args.batch_size)):
        batch_traces = traces[i:i + args.batch_size]

        if args.use_docs_only:
            batch_prompts = [gen_final_answer_docs_only_prompt(trace["question"], extract_documents_only(trace["trace"])) for trace in batch_traces]
        else:
            batch_prompts = [gen_final_answer_prompt(trace["question"], trace["trace"]) for trace in batch_traces]

        inputs = tokenizer.apply_chat_template(
            batch_prompts,
            tokenize=True,
            truncation=True,
            padding="longest",
            max_length=4096,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            batch_scores1 = outputs["preds_head1"].squeeze(-1).cpu().tolist()
            batch_scores2 = outputs["preds_head2"].squeeze(-1).cpu().tolist()

        with open(args.output_path, "a", encoding="utf-8") as f:
            for trace, score1, score2 in zip(batch_traces, batch_scores1, batch_scores2):
                trace["score1"] = score1
                trace["score2"] = score2
                f.write(json.dumps(trace) + "\n")

    print(f"Scoring completed and results saved to {args.output_path}", flush=True)
