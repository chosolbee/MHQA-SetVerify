import os
from typing import List, Dict, Any, Tuple
from vllm import SamplingParams
from .prompts import QUERY_GENERATION_PROMPT, INTERMEDIATE_ANSWER_GENERATION_PROMPT, FINAL_ANSWER_GENERATION_PROMPT


class Generator:
    def __init__(self, llm, max_gen_length=200, temperature=0.0, top_p=1.0):
        os.environ['MKL_THREADING_LAYER']='GNU'

        self.llm = llm

        self.max_gen_length = max_gen_length
        self.temperature = temperature
        self.top_p = top_p

        print("Generator loaded successfully.")

    def _process_prompts_vllm(self, prompts):
        sampling_params = SamplingParams(
            max_tokens=self.max_gen_length,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        outputs = self.llm.chat(prompts, sampling_params, use_tqdm=False)

        return [output.outputs[0].text.strip() for output in outputs]

    def batch_generate_queries(
        self,
        questions: List[Dict[str, Any]],
        traces: List[str],
        fields: Dict[str, str]
    ) -> Tuple[List[str], List[str]]:
        prompts = [
            [{
                "role": "user",
                "content": QUERY_GENERATION_PROMPT.format(
                    trace=trace or "Nothing yet",
                    question=question[fields["question"]],
                ),
            }] for question, trace in zip(questions, traces)
        ]

        outputs = self._process_prompts_vllm(prompts)

        return outputs

    def batch_generate_intermediate_answers(
        self,
        queries: List[str],
        batch_docs: List[List[Dict[str, Any]]]
    ) -> List[str]:
        prompts = [
            [{
                "role": "user",
                "content": INTERMEDIATE_ANSWER_GENERATION_PROMPT.format(
                    docs="\n\n".join([f"{doc['title']}: {doc['text']}" for doc in docs]),
                    query=query,
                ),
            }] for query, docs in zip(queries, batch_docs)
        ]

        answers = self._process_prompts_vllm(prompts)

        return answers

    def batch_generate_final_answers(
        self,
        questions: List[Dict[str, Any]],
        batch_history: List[List[Dict[str, Any]]],
        traces: List[str],
        fields: Dict[str, str]
    ) -> Tuple[List[str], List[str]]:
        prompts = [
            [{
                "role": "user",
                "content": FINAL_ANSWER_GENERATION_PROMPT.format(
                    docs="\n\n".join([f"Doc {idx}: {doc['title']}: {doc['text']}" for idx, doc in enumerate(history)]),
                    trace=trace,
                    question=question[fields["question"]],
                ),
            }] for question, history, trace in zip(questions, batch_history, traces)
        ]

        answers = self._process_prompts_vllm(prompts)

        return answers
