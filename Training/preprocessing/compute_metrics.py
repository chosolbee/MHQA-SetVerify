import os
import sys
import json
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from vllm import LLM, SamplingParams
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from pipeline.answer_generator.prompts import gen_final_answer_prompt
from pipeline.utils import compute_all_answer_metrics

tqdm.pandas()


def compute_max_cont_metrics(group):
    max_cont_ems = []
    max_cont_f1s = []
    for _, row in group.iterrows():
        future_mask = group["iter_cnt"] > row["iter_cnt"]
        if future_mask.any():
            max_cont_em = group.loc[future_mask, "em"].max()
            max_cont_f1 = group.loc[future_mask, "f1"].max()
        else:
            max_cont_em = 0.0
            max_cont_f1 = 0.0
        max_cont_ems.append(max_cont_em)
        max_cont_f1s.append(max_cont_f1)

    group = group.copy()
    group["max_cont_em"] = max_cont_ems
    group["max_cont_f1"] = max_cont_f1s
    return group


def parse_args():
    parser = argparse.ArgumentParser(description="Compute answer scores from the dataset")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for processing")
    parser.add_argument("--repeat-size", type=int, default=8, help="Number of times to repeat each trace")

    vllm_group = parser.add_argument_group("vLLM Options")
    vllm_group.add_argument("--vllm-model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID for vLLM")
    vllm_group.add_argument("--vllm-tp-size", type=int, default=1, help="Tensor parallel size for vLLM")
    vllm_group.add_argument("--vllm-quantization", type=str, help="Quantization method for vLLM")
    vllm_group.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    vllm_group.add_argument("--vllm-max-model-len", type=int, default=8192, help="Maximum model length for vLLM")

    final_answer_generator_group = parser.add_argument_group("Final Answer Generator Options")
    final_answer_generator_group.add_argument("--fag-max-gen-length", type=int, default=400, help="Maximum generation length for answer generator")
    final_answer_generator_group.add_argument("--fag-temperature", type=float, default=0.7, help="Temperature for answer generator")
    final_answer_generator_group.add_argument("--fag-top-p", type=float, default=0.9, help="Top-p sampling for answer generator")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = LLM(
        model=args.vllm_model_id,
        tensor_parallel_size=args.vllm_tp_size,
        quantization=args.vllm_quantization,
        dtype=torch.bfloat16,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.vllm_max_model_len,
    )

    sampling_params = SamplingParams(
        max_tokens=args.fag_max_gen_length,
        temperature=args.fag_temperature,
        top_p=args.fag_top_p,
    )

    print("Model and tokenizer loaded successfully.", flush=True)

    with open(args.input_path, "r", encoding="utf-8") as f:
        traces = [json.loads(line.strip()) for line in f]

    open(args.output_path, "w", encoding="utf-8").close()

    print(f"Number of prompts: {len(traces)}", flush=True)

    assert args.batch_size % args.repeat_size == 0
    effective_batch_size = args.batch_size // args.repeat_size

    for i in tqdm(range(0, len(traces), effective_batch_size)):
        batch_traces = traces[i:i+effective_batch_size]
        batch_traces_repeated = [trace for trace in batch_traces for _ in range(args.repeat_size)]
        batch_prompts_repeated = [gen_final_answer_prompt(trace["question"], trace["trace"]) for trace in batch_traces_repeated]
        batch_answers_repeated = [{
            "answer": trace["answers"][0],
            "answer_aliases": trace["answers"][1:],
        } for trace in batch_traces_repeated]

        outputs = model.chat(batch_prompts_repeated, sampling_params)
        batch_predictions_repeated = [output.outputs[0].text.strip() for output in outputs]

        em_list, f1_list = compute_all_answer_metrics(batch_answers_repeated, batch_predictions_repeated)

        avg_em_list = np.mean(np.array(em_list).reshape(-1, args.repeat_size), axis=1)
        avg_f1_list = np.mean(np.array(f1_list).reshape(-1, args.repeat_size), axis=1)

        with open(args.output_path, "a", encoding="utf-8") as f:
            for trace, em, f1 in zip(batch_traces, avg_em_list, avg_f1_list):
                trace["em"] = em
                trace["f1"] = f1
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    df = pd.DataFrame(traces)

    print("Computing 'max_cont_em' and 'max_cont_f1'...")

    df = df.sort_values(["question_id", "iter_cnt"])
    df = df.groupby("question_id", group_keys=False).progress_apply(compute_max_cont_metrics)

    print("Writing output...")

    df.to_json(args.output_path, orient="records", lines=True, force_ascii=False, mode="w")

    print(f"Processing complete. Output written to {args.output_path}")
