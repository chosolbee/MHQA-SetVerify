import os
import sys
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
from vllm import LLM, SamplingParams
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from pipeline.answer_generator.prompts import gen_final_answer_prompt
from pipeline.utils import compute_all_answer_metrics


def extract_answer(output):
    if "answer is: " in output.lower():
        idx = output.lower().find("answer is: ")
        return output[idx + len("answer is: "):].split("\n")[0].strip()
    else:
        return output


def parse_args():
    parser = argparse.ArgumentParser(description="Compute answer scores from the dataset")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for processing")
    parser.add_argument("--repeat-size", type=int, default=8, help="Number of times to repeat each trace")
    parser.add_argument("--icl-examples-path", type=str, help="Path to ICL examples (Required for docs-only mode)")
    parser.add_argument("--use-docs-only", action="store_true", help="Use only documents from trace")

    vllm_group = parser.add_argument_group("vLLM Options")
    vllm_group.add_argument("--vllm-model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID for vLLM")
    vllm_group.add_argument("--vllm-tp-size", type=int, default=1, help="Tensor parallel size for vLLM")
    vllm_group.add_argument("--vllm-quantization", type=str, help="Quantization method for vLLM")
    vllm_group.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    vllm_group.add_argument("--vllm-max-model-len", type=int, help="Maximum model length for vLLM")

    final_answer_generator_group = parser.add_argument_group("Final Answer Generator Options")
    final_answer_generator_group.add_argument("--fag-max-gen-length", type=int, default=400, help="Maximum generation length for answer generator")
    final_answer_generator_group.add_argument("--fag-temperature", type=float, default=0.0, help="Temperature for answer generator")
    final_answer_generator_group.add_argument("--fag-top-p", type=float, default=1.0, help="Top-p sampling for answer generator")

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

    if args.use_docs_only:
        with open(args.icl_examples_path, "r", encoding="utf-8") as f:
            icl_examples = f.read()
            icl_examples = "\n".join([line for line in icl_examples.split("\n") if not line.startswith("# METADATA")])

    open(args.output_path, "w", encoding="utf-8").close()

    print(f"Number of prompts: {len(traces)}", flush=True)

    assert args.batch_size % args.repeat_size == 0
    effective_batch_size = args.batch_size // args.repeat_size

    for i in tqdm(range(0, len(traces), effective_batch_size)):
        batch_traces = traces[i:i+effective_batch_size]
        batch_traces_repeated = [trace for trace in batch_traces for _ in range(args.repeat_size)]

        batch_answers_repeated = [{
            "answer": trace["answers"][0],
            "answer_aliases": trace["answers"][1:],
        } for trace in batch_traces_repeated]

        if args.use_docs_only:
            batch_prompts_repeated = [
                icl_examples + "\n\n" + \
                "\n\n".join([f"Wikipedia Title: {doc['title']}\n{doc['text']}" for doc in trace["history"]]) + \
                "\n\n" + "Q: " + trace["question"] + "\n" + "A: "
                for trace in batch_traces_repeated
            ]
            outputs = model.generate(batch_prompts_repeated, sampling_params, use_tqdm=False)
            batch_predictions_repeated = [extract_answer(output.outputs[0].text.strip()) for output in outputs]
        else:
            batch_prompts_repeated = [gen_final_answer_prompt(trace["question"], trace["trace"]) for trace in batch_traces_repeated]
            outputs = model.chat(batch_prompts_repeated, sampling_params, use_tqdm=False)
            batch_predictions_repeated = [output.outputs[0].text.strip() for output in outputs]

        fields = {
            "answer": "answer",
            "answer_aliases": "answer_aliases",
        }

        em_list, f1_list, acc_list = compute_all_answer_metrics(batch_answers_repeated, batch_predictions_repeated, fields)

        avg_em_list = np.mean(np.array(em_list).reshape(-1, args.repeat_size), axis=1)
        avg_f1_list = np.mean(np.array(f1_list).reshape(-1, args.repeat_size), axis=1)
        avg_acc_list = np.mean(np.array(acc_list).reshape(-1, args.repeat_size), axis=1)

        with open(args.output_path, "a", encoding="utf-8") as f:
            for trace, em, f1, acc in zip(batch_traces, avg_em_list, avg_f1_list, avg_acc_list):
                trace["em"] = em
                trace["f1"] = f1
                trace["acc"] = acc
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    print(f"Processing complete. Output written to {args.output_path}")
