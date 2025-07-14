import os
import sys
import json
import argparse
from tqdm import tqdm
import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from pipeline.answer_generator.prompts import gen_final_answer_prompt

tqdm.pandas()


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def compute_max_cont_prob(group):
    max_cont_probs = []
    for _, row in group.iterrows():
        future_mask = group["iter_cnt"] > row["iter_cnt"]
        if future_mask.any():
            max_cont_prob = group.loc[future_mask, "prob"].max()
        else:
            max_cont_prob = 0.0
        max_cont_probs.append(max_cont_prob)

    group = group.copy()
    group["max_cont_prob"] = max_cont_probs
    return group


def parse_args():
    parser = argparse.ArgumentParser(description="Extract partial traces from the dataset")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID for computing scores")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for calculating probabilities")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    rank, world_size, local_rank = setup_distributed()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        tp_plan="auto",
    )

    print("Model and tokenizer loaded successfully.", flush=True)

    with open(args.input_path, "r", encoding="utf-8") as f:
        traces = [json.loads(line.strip()) for line in f]

    if rank == 0:
        open(args.output_path, "w", encoding="utf-8").close()

    if world_size > 1:
        dist.barrier()

    prompts = []
    completions = []
    for trace in traces:
        question = trace["question"]
        trace_text = trace["trace"]
        chat = gen_final_answer_prompt(question, trace_text)
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        answer = trace["answers"][0]
        completions.append(answer)

    print(f"Number of prompts: {len(prompts)}", flush=True)
    print(f"Number of completions: {len(completions)}", flush=True)

    print("Computing 'prob'...")

    batch_size = 8
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        batch_completions = completions[i:i+batch_size]

        prompt_token_counts = [len(tokenizer.encode(p)) for p in batch_prompts]

        combined_texts = [p + c for p, c in zip(batch_prompts, batch_completions)]
        inputs = tokenizer(combined_texts, padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        del outputs
        logits /= args.temperature

        total_items = len(logits)
        items_per_process = total_items // world_size
        start_idx = rank * items_per_process
        if rank == world_size - 1:
            end_idx = total_items
        else:
            end_idx = start_idx + items_per_process

        for j in range(start_idx, end_idx):
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

            trace = traces[i + j]
            trace["log_prob"] = log_p.item()
            trace["prob"] = p.item()

            with open(args.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")

            del completion_logits, completion_token_ids, log_probs, completion_log_probs

        del logits
        torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        df = pd.DataFrame(traces)

        print("Computing 'max_cont_prob'...")

        df = df.sort_values(["question_id", "iter_cnt"])
        df = df.groupby("question_id", group_keys=False).progress_apply(compute_max_cont_prob)

        print("Writing output...")

        df.to_json(args.output_path, orient="records", lines=True, force_ascii=False, mode="w")

        print(f"Processing complete. Output written to {args.output_path}")

    if world_size > 1:
        dist.barrier()

    cleanup_distributed()
