import os
import sys
import json
import argparse
from tqdm import tqdm
import pandas as pd
import torch
from transformers import set_seed
from vllm import LLM
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from pipeline.modules import AsyncOpenAIConfig
from pipeline.stop_decider import StopDecider

COLUMNS = ["em", "f1", "acc", "retrieval_em", "retrieval_precision", "retrieval_recall", "retrieval_f1", "iter_cnt"]

pd.set_option('display.max_columns', None)


def pick_values(subdf):
    over = subdf[subdf["decision"] == "STOP"]
    chosen = over.iloc[0] if not over.empty else subdf.iloc[-1]
    return chosen[COLUMNS + ["num_hops"]]


def parse_args():
    parser = argparse.ArgumentParser(description="Test LLM stop decider")

    vllm_group = parser.add_argument_group("vLLM Options")
    vllm_group.add_argument("--vllm-model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID for vLLM")
    vllm_group.add_argument("--vllm-tp-size", type=int, default=1, help="Tensor parallel size for vLLM")
    vllm_group.add_argument("--vllm-quantization", type=str, help="Quantization method for vLLM")
    vllm_group.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    vllm_group.add_argument("--vllm-max-model-len", type=int, help="Maximum model length for vLLM")

    openai_group = parser.add_argument_group("OpenAI Options")
    openai_group.add_argument("--openai-model-id", type=str, default="gpt-4o-mini-2024-07-18", help="Model ID for OpenAI")
    openai_group.add_argument("--openai-max-retries", type=int, default=1, help="Maximum retries for OpenAI requests")
    openai_group.add_argument("--openai-batch-timeout", type=int, default=600, help="Batch timeout for OpenAI requests")
    openai_group.add_argument("--openai-total-timeout", type=int, default=60, help="Total timeout for OpenAI requests")
    openai_group.add_argument("--openai-connect-timeout", type=int, default=10, help="Connection timeout for OpenAI requests")
    openai_group.add_argument("--openai-max-keepalive-connections", type=int, default=10, help="Maximum keepalive connections for OpenAI")
    openai_group.add_argument("--openai-max-connections", type=int, default=20, help="Maximum connections for OpenAI")
    openai_group.add_argument("--openai-max-concurrent", type=int, default=3, help="Maximum concurrent requests for OpenAI")

    stop_decider_group = parser.add_argument_group("Stop Decider Options")
    stop_decider_group.add_argument("--sd-max-gen-length", type=int, default=200, help="Maximum generation length for stop decider")
    stop_decider_group.add_argument("--sd-temperature", type=float, default=0.0, help="Temperature for stop decider")
    stop_decider_group.add_argument("--sd-top-p", type=float, default=1.0, help="Top-p sampling for stop decider")
    stop_decider_group.add_argument("--sd-provider", type=str, default="vllm", choices=["vllm", "openai"], help="Provider for stop decider")

    main_group = parser.add_argument_group("Main Options")
    main_group.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")
    main_group.add_argument("--batch-size", type=int, default=128, help="Batch size for processing")
    main_group.add_argument("--use-docs-only", action="store_true", help="Use only documents from trace")
    main_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)

    if args.sd_provider == "vllm":
        llm = LLM(
            model=args.vllm_model_id,
            tensor_parallel_size=args.vllm_tp_size,
            quantization=args.vllm_quantization,
            dtype=torch.bfloat16,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=args.vllm_max_model_len,
        )
    elif args.sd_provider == "openai":
        llm = AsyncOpenAIConfig(
            model_id=args.openai_model_id,
            max_retries=args.openai_max_retries,
            batch_timeout=args.openai_batch_timeout,
            total_timeout=args.openai_total_timeout,
            connect_timeout=args.openai_connect_timeout,
            max_keepalive_connections=args.openai_max_keepalive_connections,
            max_connections=args.openai_max_connections,
            max_concurrent=args.openai_max_concurrent,
        )
    else:
        raise ValueError(f"Unknown provider: {args.sd_provider}")

    stop_decider = StopDecider(
        llm=llm,
        max_gen_length=args.sd_max_gen_length,
        temperature=args.sd_temperature,
        top_p=args.sd_top_p,
        provider=args.sd_provider,
        use_docs_only=args.use_docs_only,
    )

    with open(args.input_path, "r", encoding="utf-8") as f:
        traces = [json.loads(line.strip()) for line in f]

    fields = {"question": "question"}

    for i in tqdm(range(0, len(traces), args.batch_size)):
        batch_traces = traces[i:i + args.batch_size]
        batch_questions = [{"question": trace["question"]} for trace in batch_traces]
        if args.use_docs_only:
            batch_trace_texts = ["\n".join(f"Document: {doc['title']}: {doc['text']}" for doc in trace["history"]) for trace in batch_traces]
        else:
            batch_trace_texts = [trace["trace"] for trace in batch_traces]

        stop_decisions = stop_decider.batch_decide(batch_questions, batch_trace_texts, fields)

        for trace, decision in zip(batch_traces, stop_decisions):
            trace["decision"] = decision

    df = pd.DataFrame(traces)
    df = df.sort_values(["question_id", "iter_cnt"])

    result = df.groupby("question_id", sort=False).apply(pick_values).reset_index()

    print("\nDetailed summary:")
    summary = result.groupby('num_hops')[COLUMNS].agg(['mean', 'count'])
    print(summary)

    overall_averages = result[COLUMNS].mean()
    print("\nOverall averages:")
    print(overall_averages)
