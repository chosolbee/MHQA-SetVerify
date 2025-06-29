import os
import sys
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pipeline.answer_generator.prompts import (
    FINAL_ANSWER_GENERATION_SYSTEM_PROMPT,
    FINAL_ANSWER_GENERATION_USER_PROMPT,
)


def gen_final_answer_prompt(question: str, trace: str, tokenizer) -> str:
    chat = [
        {
            "role": "system",
            "content": FINAL_ANSWER_GENERATION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "Main question: " + question.strip() + "\n\n" + trace.strip() + "\n\n" + FINAL_ANSWER_GENERATION_USER_PROMPT,
        },
    ]

    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract partial traces from the dataset")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for calculating probabilities")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        tp_plan="auto",
    ).to(device)

    print("Model and tokenizer loaded successfully.", flush=True)

    with open(args.input_path, "r", encoding="utf-8") as f:
        traces = [json.loads(line.strip()) for line in f]

    open(args.output_path, "w", encoding="utf-8").close()

    prompts = []
    completions = []
    for trace in traces:
        question = trace["question"]
        trace_text = trace["trace"]
        prompt = gen_final_answer_prompt(question, trace_text, tokenizer)
        prompts.append(prompt)
        answer = trace["answers"][0]
        completions.append(answer)
    
    print(f"Number of prompts: {len(prompts)}", flush=True)
    print(f"Number of completions: {len(completions)}", flush=True)

    records = []
    batch_size = 8
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        batch_completions = completions[i:i+batch_size]

        prompt_token_counts = [len(tokenizer.encode(p)) for p in batch_prompts]

        combined_texts = [p + c for p, c in zip(batch_prompts, batch_completions)]
        inputs = tokenizer(combined_texts, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits / args.temperature

        all_total_log_probs = []
        all_probs = []
        for j in range(min(batch_size, len(batch_prompts))):
            prompt_len = prompt_token_counts[j]

            completion_logits = logits[j, prompt_len-1:-1, :]
            completion_token_ids = inputs["input_ids"][j, prompt_len:]

            actual_completion_len = torch.sum(inputs["attention_mask"][j, prompt_len:]).item()

            completion_logits = completion_logits[:actual_completion_len]
            completion_token_ids = completion_token_ids[:actual_completion_len]

            log_probs = torch.nn.functional.log_softmax(completion_logits, dim=-1)
            completion_log_probs = torch.gather(log_probs, 1, completion_token_ids.unsqueeze(-1)).squeeze(-1)

            log_p = completion_log_probs.sum()
            p = log_p.exp().clamp(1e-12, 1-1e-12)

            with open(args.output_path, "a", encoding="utf-8") as f:
                trace = traces[i + j]
                trace["log_prob"] = log_p.item()
                trace["prob"] = p.item()
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")

            del completion_logits, completion_token_ids

        del inputs, outputs, logits
        torch.cuda.empty_cache()
