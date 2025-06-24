import os
import re
import torch
from typing import List, Dict, Any, Tuple
from vllm import LLM, SamplingParams
from .prompts import (
    INTERMEDIATE_ANSWER_GENERATION_SYSTEM_PROMPT,
    FINAL_ANSWER_GENERATION_SYSTEM_PROMPT,
    FINAL_ANSWER_GENERATION_USER_PROMPT,
)


class AnswerGenerator:
    def __init__(self, llm, max_gen_length=400, temperature=0.7, top_p=0.9):
        os.environ["MKL_THREADING_LAYER"] = "GNU"

        self.llm = llm
        self.tokenizer = llm.get_tokenizer()

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_gen_length,
        )

        print("Answer Generator initialized successfully.")

    def _gen_intermediate_answer_prompt(self, trace: str) -> str:
        """Generate prompt for intermediate answer generation"""
        chat = [
            {
                "role": "system",
                "content": INTERMEDIATE_ANSWER_GENERATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": trace.strip(),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        # ### token수 확인
        # n_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        # print(f"[TOKENS][AG_Inter] {n_tokens}")

        return prompt

    def _gen_final_answer_prompt(self, question: str, trace: str) -> str:
        """Generate prompt for final answer generation"""
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

        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ### token수 확인
        # n_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        # print(f"[TOKENS][AG_final] {n_tokens}")

        return prompt

    def batch_generate_intermediate_answers(self, queries: List[str], docs: List[str]) -> List[str]:
        prompts = [
            self._gen_intermediate_answer_prompt(f"Question: {query}\nDocument: {doc}")
            for query, doc in zip(queries, docs)
        ]
        outputs = self.llm.generate(prompts, self.sampling_params)
        answers = [output.outputs[0].text.strip() for output in outputs]

        return answers

    def batch_generate_final_answers(self, questions: List[str], traces: List[str]) -> List[str]:
        prompts = [
            self._gen_final_answer_prompt(question["question"], trace)
            for question, trace in zip(questions, traces)
        ]
        outputs = self.llm.generate(prompts, self.sampling_params)
        answers = [output.outputs[0].text.strip() for output in outputs]
        new_traces = [
            trace + "\nFinal answer: " + answer
            for trace, answer in zip(traces, answers)
        ]

        return new_traces, answers

    def generate_intermediate_answer(self, query: Dict[str, Any], doc: str) -> Tuple[str, str]:
        new_traces, answers = self.batch_generate_intermediate_answers([query], [doc])
        return new_traces[0], answers[0]

    def generate_final_answer(self, question: Dict[str, Any], trace: str) -> Tuple[str, str]:
        new_traces, answers = self.batch_generate_final_answers([question], [trace])
        return new_traces[0], answers[0]


def test():
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        quantization=None,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )

    answer_generator = AnswerGenerator(
        llm=llm,
        max_gen_length=200,
        temperature=0.7,
        top_p=0.9,
    )

    print("\n=== Testing Intermediate Answer Generation ===")
    queries = ["Where did Peter Kern die?"]
    docs = ["Peter Kern (American businessman) Peter Kern (October 31, 1835 – October 28, 1907) was a German-born American businessman and politician active in Knoxville, Tennessee, USA, in the late 19th and early 20th centuries."]

    new_traces, answers = answer_generator.batch_generate_intermediate_answers(queries, docs)
    for query, doc, new_trace, answer in zip(queries, docs, new_traces, answers):
        print(f"Query: {query}")
        print("-" * 50)
        print(f"Document: {doc}")
        print("-" * 50)
        print(f"Generated Answer: {answer}")
        print("-" * 50)
        print(f"New Trace: {new_trace}")
        print("=" * 50)

    print("\n=== Testing Final Answer Generation ===")
    questions = [{"question": "What county is the city where Peter Kern died in?"}]
    traces = ["Follow up: Where did Peter Kern die?\nDocument: Peter Kern (American businessman) Peter Kern (October 31, 1835 – October 28, 1907) was a German-born American businessman and politician active in Knoxville, Tennessee, USA, in the late 19th and early 20th centuries.\nIntermediate answer: Peter Kern died in Knoxville, Tennessee.\nFollow up: In what county is Knoxville, Tennessee located?\nDocument: Knoxville is a city in the U.S. state of Tennessee, and the county seat of Knox County.\nIntermediate answer: Knoxville is located in Knox County."]

    new_traces, answers = answer_generator.batch_generate_final_answers(questions, traces)
    for question, trace, new_trace, answer in zip(questions, traces, new_traces, answers):
        print(f"Question: {question['question']}")
        print("-" * 50)
        print(f"Original Trace: {trace}")
        print("-" * 50)
        print(f"Generated Answer: {answer}")
        print("-" * 50)
        print(f"New Trace: {new_trace}")
        print("=" * 50)

    print("\nTest completed.")


if __name__ == "__main__":
    test()
