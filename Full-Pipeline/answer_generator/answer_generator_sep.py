import os
import re
import torch
from typing import List, Dict, Any, Tuple
from vllm import LLM, SamplingParams
from .prompts import INTERMEDIATE_ANSWER_GENERATION_PROMPT, FINAL_ANSWER_GENERATION_PROMPT

class AnswerGenerator:
    def __init__(self, llm, max_gen_length=200, temperature=0.7, top_p=0.9):
        os.environ["MKL_THREADING_LAYER"] = "GNU"

        self.llm = llm
        self.tokenizer = llm.get_tokenizer()

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_gen_length,
        )

        print("Answer Generator initialized successfully.")

    def _extract_answer(self, text: str, final: bool) -> str:
        prefix = "Final answer:" if final else "Intermediate answer:"
        for line in text.strip().splitlines():
            line = line.strip()
            if line.startswith(prefix):
                return line[len(prefix):].strip()
        return text.strip()

    def _gen_intermediate_answer_prompt(self, question: str, trace: str) -> str:
        """Generate prompt for intermediate answer generation"""
        chat = [
            {
                "role": "system",
                "content": INTERMEDIATE_ANSWER_GENERATION_PROMPT,
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
                "content": FINAL_ANSWER_GENERATION_PROMPT,
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
        
        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ### token수 확인
        # n_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        # print(f"[TOKENS][AG_final] {n_tokens}")
        
        return prompt

    def batch_generate_answers(self, questions: List[Dict[str, Any]], traces: List[str], 
                              is_final: List[bool]) -> Tuple[List[str], List[str]]:
        prompts = []
        
        for question, trace, final in zip(questions, traces, is_final):
            if final:
                prompt = self._gen_final_answer_prompt(question["question"], trace)
            else:
                prompt = self._gen_intermediate_answer_prompt(question["question"], trace)
            prompts.append(prompt)
        
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        new_traces = []
        answers = []
        
        for trace, output, final in zip(traces, outputs, is_final):
            answer = output.outputs[0].text
            answer = self._extract_answer(answer, final)
            
            tag = "Final answer:" if final else "Intermediate answer:"
            new_traces.append(f"{trace}\n{tag} {answer}")
            answers.append(answer)
                    
        return new_traces, answers

    def generate_intermediate_answer(self, question: Dict[str, Any], trace: str) -> Tuple[str, str]:
        new_traces, answers = self.batch_generate_answers([question], [trace], [False])
        return new_traces[0], answers[0]
    
    def generate_final_answer(self, question: Dict[str, Any], trace: str) -> Tuple[str, str]:
        new_traces, answers = self.batch_generate_answers([question], [trace], [True])
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

    questions = [{"question": "What county is the city where Peter Kern died in?"}]
    
    print("\n=== Testing Intermediate Answer Generation ===")
    traces = ["Follow up: Where did Peter Kern die?\nDocument: Peter Kern (American businessman) Peter Kern (October 31, 1835 – October 28, 1907) was a German-born American businessman and politician active in Knoxville, Tennessee, USA, in the late 19th and early 20th centuries."]
    
    new_traces, answers = answer_generator.batch_generate_answers(questions, traces, [False])
    for question, trace, new_trace, answer in zip(questions, traces, new_traces, answers):
        print(f"Question: {question['question']}")
        print("-" * 50)
        print(f"Original Trace: {trace}")
        print("-" * 50)
        print(f"Generated Answer: {answer}")
        print("-" * 50)
        print(f"New Trace: {new_trace}")
        print("=" * 50)
    
    print("\n=== Testing Final Answer Generation ===")
    traces = ["Follow up: Where did Peter Kern die?\nDocument: Peter Kern (American businessman) Peter Kern (October 31, 1835 – October 28, 1907) was a German-born American businessman and politician active in Knoxville, Tennessee, USA, in the late 19th and early 20th centuries.\nIntermediate answer: Peter Kern died in Knoxville, Tennessee.\nFollow up: In what county is Knoxville, Tennessee located?\nDocument: Knoxville is a city in the U.S. state of Tennessee, and the county seat of Knox County.\nIntermediate answer: Knoxville is located in Knox County."]

    new_traces, answers = answer_generator.batch_generate_answers(questions, traces, [True])
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