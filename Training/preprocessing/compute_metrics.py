import os
import sys
import json
import argparse
from typing import List, Tuple
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from pipeline.answer_generator.prompts import gen_final_answer_prompt
from pipeline.utils import compute_answer_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Compute answer scores from the dataset")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for processing")

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

    print(f"Number of prompts: {len(traces)}", flush=True)

    open(args.output_path, "w", encoding="utf-8").close()

    predictions = []
    for i in tqdm(range(0, len(traces), args.batch_size)):
        batch_traces = traces[i:i+args.batch_size]
        batch_prompts = [gen_final_answer_prompt(trace["question"], trace["trace"]) for trace in batch_traces]
        batch_answers = [{
            "answer": trace["answers"][0],
            "answer_aliases": trace["answers"][1:],
        } for trace in batch_traces]

        outputs = model.chat(batch_prompts, sampling_params)
        batch_predictions = [output.outputs[0].text.strip() for output in outputs]

        em_list, f1_list = compute_answer_metrics(batch_answers, batch_predictions)

        with open(args.output_path, "a", encoding="utf-8") as f:
            for trace, prediction, em, f1 in zip(batch_traces, batch_predictions, em_list, f1_list):
                trace["prediction"] = prediction
                trace["em"] = em
                trace["f1"] = f1
                f.write(json.dumps(trace) + "\n")
