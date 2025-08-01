import os
import sys
import json
import argparse
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    set_seed,
)
from peft import PeftModelForSequenceClassification
from ..utils import extract_documents_only, convert_chat_to_text
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from pipeline.answer_generator.prompts import gen_final_answer_prompt, gen_final_answer_docs_only_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Compute answer scores from the dataset")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for processing")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization for the model")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for the model")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model ID for computing scores")
    parser.add_argument("--max-length", type=int, default=4096, help="Max Length of Inputs")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision for model")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the trained checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use-docs-only", action="store_true", help="Use only documents from trace")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)

    nf4_config = None
    if args.use_4bit:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id if args.use_lora else args.checkpoint_path,
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
        num_labels=1,
    )

    if args.use_lora:
        model = PeftModelForSequenceClassification.from_pretrained(
            model,
            args.checkpoint_path,
            device_map="auto"
        )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
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

        has_chat_template = (
            hasattr(tokenizer, 'chat_template') and 
            tokenizer.chat_template is not None
        )

        if has_chat_template:  # Decoder
            inputs = tokenizer.apply_chat_template(
                batch_prompts,
                tokenize=True,
                truncation=True,
                padding="longest",
                max_length=args.max_length,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
        else:  # Encoder
            batch_texts = [convert_chat_to_text(prompt, tokenizer, args.use_docs_only) for prompt in batch_prompts]
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding="longest",
                max_length=args.max_length,
                return_tensors="pt"
            ).to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1)
            batch_scores = torch.sigmoid(logits).cpu().tolist()

        with open(args.output_path, "a", encoding="utf-8") as f:
            for trace, score in zip(batch_traces, batch_scores):
                trace["score"] = score
                f.write(json.dumps(trace) + "\n")

    print(f"Scoring completed and results saved to {args.output_path}", flush=True)
