import os
import sys
import asyncio
from .prompts import gen_retriever_query_prompt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from modules import AsyncOpenAIProcessor


class QueryGenerator:
    def __init__(self, llm, max_gen_length=200, temperature=0.7, top_p=0.9, provider="vllm"):
        os.environ['MKL_THREADING_LAYER']='GNU'

        self.llm = llm

        self.max_gen_length = max_gen_length
        self.temperature = temperature
        self.top_p = top_p

        self.provider = provider

        print(f"Query Generator - {self.provider} loaded successfully.")

    def _process_prompts_vllm(self, prompts):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=self.max_gen_length,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        outputs = self.llm.chat(prompts, sampling_params)

        return [output.outputs[0].text.strip() for output in outputs]

    async def _process_prompts_openai_async(self, prompts):
        async with AsyncOpenAIProcessor(self.llm) as processor:
            return await processor.process_prompts_async(
                prompts,
                max_gen_length=self.max_gen_length,
                temperature=self.temperature,
                top_p=self.top_p,
            )

    def extract_query(self, text):
        query = text.strip().splitlines()[0].strip()
        if query.lower().startswith("follow up: ") or query.lower().startswith("follow-up: "):
            query = query[11:].strip()
        return query

    def batch_generate(self, questions: list[str], traces: list[str]) -> tuple[list[str], list[str]]:
        prompts = [
            gen_retriever_query_prompt(question["question"], trace)
            for question, trace in zip(questions, traces)
        ]

        if self.provider == "vllm":
            outputs = self._process_prompts_vllm(prompts)
        elif self.provider == "openai":
            outputs = asyncio.run(self._process_prompts_openai_async(prompts))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        new_traces = []
        queries = []
        for trace, output in zip(traces, outputs):
            gen_text = output
            query = self.extract_query(gen_text)
            new_traces.append(trace + "\nFollow up: " + query.strip())
            queries.append(query)

        return new_traces, queries


def test(llm, provider):
    query_generator = QueryGenerator(
        llm=llm,
        max_gen_length=2048,
        temperature=0.7,
        top_p=0.9,
        provider="provider",
    )

    questions = [{"question": "What county is the city where Peter Kern died in?"}]
    traces = ["Follow up: Where did Peter Kern die?\nDocument: Peter Kern (American businessman) Peter Kern (October 31, 1835 – October 28, 1907) was a German-born American businessman and politician active in Knoxville, Tennessee, USA, in the late 19th and early 20th centuries. He is best known as the founder of the confections company that eventually evolved into Kern's Bakery, a brand still marketed in the Knoxville area. The company's former confectionery and ice cream parlor, now called the Mall Building (or Oliver Hotel), still dominates the southwest corner of Market Square. Kern served as Knoxville's mayor from 1890 until 1892. Kern was born in Zwingenberg (near Heidelberg) in Germany\nIntermediate answer: Peter Kern died in Knoxville, Tennessee."]
    # query should be like -> Follow up: In what county is Knoxville, Tennessee located?

    new_traces, queries = query_generator.batch_generate(questions, traces)

    for nt, q in zip(new_traces, queries):
        print("New Trace:", nt)
        print("-" * 50)
        print("Generated Query:", q)
        print("-" * 50)
    print("Test completed.")

    new_traces, queries = query_generator.batch_generate(questions, traces)

    for nt, q in zip(new_traces, queries):
        print("New Trace:", nt)
        print("-" * 50)
        print("Generated Query:", q)
        print("-" * 50)
    print("Test completed.")


if __name__ == "__main__":
    # Test vLLM
    import torch
    from vllm import LLM

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        quantization=None,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )

    test(llm, "vllm")

    # Test OpenAI
    from modules import AsyncOpenAIConfig

    llm = AsyncOpenAIConfig(
        model_id="gpt-4o-mini-2024-07-18",
        max_retries=1,
        timeout=60,
    )

    test(llm, "openai")
