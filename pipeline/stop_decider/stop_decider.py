import os
import sys
import asyncio
from typing import List, Dict, Any
from .prompts import STOP_DECISION_SYSTEM_PROMPT
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from modules import AsyncOpenAIProcessor

class StopDecider:
    def __init__(self, llm, max_gen_length=200, temperature=0.3, top_p=0.9, provider="vllm"):
        os.environ['MKL_THREADING_LAYER']='GNU'

        self.llm = llm

        self.max_gen_length = max_gen_length
        self.temperature = temperature
        self.top_p = top_p

        self.provider = provider

        if self.provider == "vllm" and llm is not None:
            self.tokenizer = llm.get_tokenizer()

        print(f"Stop Decider - {self.provider} initialized successfully.")

    def _gen_stop_decision_prompt(self, question, trace):
        chat = [
            {
                "role": "system",
                "content": STOP_DECISION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question.strip(),
            },
            {
                "role": "assistant",
                "content": trace.strip(),
            },
        ]

        if self.provider == "vllm":
            prompt = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
        else:
            return chat

    def _process_prompts_vllm(self, prompts):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=self.max_gen_length,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        return [output.outputs[0].text.strip() for output in outputs]

    async def _process_prompts_openai_async(self, prompts):
        async with AsyncOpenAIProcessor(self.llm) as processor:
            return await processor.process_prompts_async(
                prompts,
                max_gen_length=self.max_gen_length,
                temperature=self.temperature,
                top_p=self.top_p,
            )

    def extract_decision(self, text, log_trace=False):
        text_upper = text.strip().upper()

        for line in text.strip().splitlines():
            if line.upper().startswith("DECISION:"):
                decision_part = line[9:].strip().upper()
                if "STOP" in decision_part:
                    if log_trace:
                        print("| STOP/CONTINUE: <STOP>")
                    return "STOP"
                elif "CONTINUE" in decision_part:
                    if log_trace:
                        print("| STOP/CONTINUE: <CONTINUE>")
                    return "CONTINUE"

        # Fallback
        if "STOP" in text_upper and "CONTINUE" not in text_upper:
            if log_trace:
                print("| STOP/CONTINUE: <STOP>")
            return "STOP"
        elif "CONTINUE" in text_upper:
            if log_trace:
                print("| STOP/CONTINUE: <CONTINUE>") 
            return "CONTINUE"
        else:
            if log_trace:
                print("| STOP/CONTINUE: <DEFAULT CONTINUE>")
                print(f" DECISION: {text.strip()}")
            return "CONTINUE"

    def batch_decide(
            self,
            questions: List[Dict[str, Any]],
            traces: List[str],
            fields: Dict[str, str],
            log_trace: bool = False
        ) -> List[str]:
        if self.provider == "nostop":
            decisions = []
            for question in questions:
                if log_trace:
                    qid = question[fields["id"]]
                    print(f"| NOSTOP CONTINUE: {qid} - Always continue until max iterations")
                decision = "CONTINUE"
                decisions.append(decision)
            return decisions

        prompts = [self._gen_stop_decision_prompt(question[fields["question"]], trace) for question, trace in zip(questions, traces)]

        if self.provider == "vllm":
            outputs = self._process_prompts_vllm(prompts)
        elif self.provider == "openai":
            outputs = asyncio.run(self._process_prompts_openai_async(prompts))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        decisions = []
        for output in outputs:
            decision = self.extract_decision(output, log_trace)
            decisions.append(decision)

        return decisions


def test(llm, provider):
    stop_decider = StopDecider(
        llm=llm,
        max_gen_length=50,
        temperature=0.3,
        top_p=0.9,
        provider=provider,
    )

    questions = [{"question": "What county is the city where Peter Kern died in?"}]

    # continue
    # traces = ["Query: Where did Peter Kern die?\nDocument: Peter Kern (American businessman) Peter Kern (October 31, 1835 – October 28, 1907) was a German-born American businessman and politician active in Knoxville, Tennessee, USA, in the late 19th and early 20th centuries. He is best known as the founder of the confections company that eventually evolved into Kern's Bakery, a brand still marketed in the Knoxville area. The company's former confectionery and ice cream parlor, now called the Mall Building (or Oliver Hotel), still dominates the southwest corner of Market Square. Kern served as Knoxville's mayor from 1890 until 1892. Kern was born in Zwingenberg (near Heidelberg) in Germany\nIntermediate answer: Peter Kern died in Knoxville, Tennessee."]

    # stop
    traces = ["Query: Where did Peter Kern die?\nDocument: Peter Kern (American businessman) Peter Kern (October 31, 1835 – October 28, 1907) was a German-born American businessman and politician active in Knoxville, Tennessee, USA, in the late 19th and early 20th centuries. He is best known as the founder of the confections company that eventually evolved into Kern's Bakery, a brand still marketed in the Knoxville area. The company's former confectionery and ice cream parlor, now called the Mall Building (or Oliver Hotel), still dominates the southwest corner of Market Square. Kern served as Knoxville's mayor from 1890 until 1892. Kern was born in Zwingenberg (near Heidelberg) in Germany.\nIntermediate answer: Peter Kern died in Knoxville, Tennessee.\nQuery: In what county is Knoxville, Tennessee located?\nDocument: Knoxville is a city in the U.S. state of Tennessee, and the county seat of Knox County.\nIntermediate answer: Knoxville is located in Knox County."]

    fields = {"question": "question"}

    decisions = stop_decider.batch_decide(questions, traces, fields)
    for question, trace, decision in zip(questions, traces, decisions):
        print(f"Question: {question['question']}")
        print(f"Trace: {trace}")
        print(f"Decision: {decision}")
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
    from modules import OpenAIConfig

    llm = OpenAIConfig(
        model_id="gpt-4o-mini-2024-07-18",
        max_retries=1,
        timeout=60,
    )

    test(llm, "openai")
