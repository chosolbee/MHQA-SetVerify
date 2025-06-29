import os
import re
import torch
from typing import List
from vllm import LLM, SamplingParams
from .prompts import ANSWER_SYSTEM_PROMPT

class AnswerGenerator:
    def __init__(self, llm, max_gen_length=200, temperature=0.7, top_p=0.9):
        os.environ["MKL_THREADING_LAYER"] = "GNU"

        self.llm = llm

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_gen_length,
        )

    def _build_prompt(self, question: str, passages: List[dict]) -> str:
        system_prompt = ANSWER_SYSTEM_PROMPT
        user_prompt = "Question: " + question + "\n"
        for idx, p in enumerate(passages, start=1):
            user_prompt += f"Passage {idx}: {p['text']}\n"
        prompt = [
            {
                "role": "system",
                "content":  system_prompt.strip(),
            },
            {
                "role": "user",
                "content": user_prompt.strip(),
            },
        ]
        return prompt

    def batch_answer(self, questions: List[dict], batch_history: List[List[dict]]) -> List[str]:
        prompts = [
            self._build_prompt(question["question"], passages)
            for question, passages in zip(questions, batch_history)
        ]
        outputs = self.llm.chat(prompts, self.sampling_params)

        clean_texts = [self.extract_answer(output.outputs[0].text) for output in outputs]
        return clean_texts

    def extract_answer(self, generated_text: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
        if match:
            extracted = match.group(1)
        else:
            extracted = re.sub(r'</s>|</answer>|<answer>', '', generated_text)

        return extracted.strip()


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
        max_gen_length=2048,
        temperature=0.7,
        top_p=0.9,
    )

    questions = [
        {"question": "Who does the Nothing Suits Me Like a Suit artist play in Batman Under the Red Hood?"},
        {"question": "What is the capital of the county that shares a border with the county where WAPL is licensed to broadcast?"},
        {"question": "Who was the British general in the Battle of the city where Toussaint's performer was born?"},
    ]

    batch_history = [
        [
            {"text": "Nothing Suits Me Like a Suit: 'Nothing Suits Me Like a Suit' is a song performed by Neil Patrick Harris and the cast of the comedy series 'How I Met Your Mother' from the 100th episode 'Girls Versus Suits'. Carter Bays and Craig Thomas were nominated for the Primetime Emmy Award for Outstanding Original Music and Lyrics for writing the song."},
            {"text": "Batman: Under the Red Hood: Batman: Under the Red Hood is a 2010 American animated superhero direct - to - video film produced by Warner Bros. Animation and released by Warner Home Video. It is the eighth feature in the DC Universe Animated Original Movies series. It was released on July 27, 2010. The film stars Bruce Greenwood as Bruce Wayne / Batman, Jensen Ackles as the Red Hood / Jason Todd, John DiMaggio as the Joker, Neil Patrick Harris as Nightwing / Dick Grayson, Jason Isaacs as Ra's al Ghul, and Wade Williams as Black Mask. The screenplay was written by Judd Winick, who also wrote the ``Under the Hood ''run in the monthly Batman comic."},
        ],
        [
            {"text": "WAPL: WAPL (105.7 FM) is a classic rock formatted radio station licensed to Appleton, Wisconsin, that serves the Green Bay and Appleton-Oshkosh areas. The station is owned by Woodward Communications, and has studios on College Avenue in Appleton, with transmitting facilities located near the WGBA Tower west of unincorporated Shirley in the Town of Glenmore in southeastern Brown County."},
            {"text": "Pulaski High School: Pulaski High School is a public high school in Pulaski, Wisconsin, in Brown County, Wisconsin (school district also serves parts of Shawano, Outagamie and Oconto counties), that serves students in grades 9 through 12. Its mascot is the Red Raider."},
            {"text": "Jerome Quinn: Born in Green Bay, Wisconsin, Quinn was a realtor and served on the Green Bay Common Council, the Brown County, Wisconsin Board of Supervisors, the local Board of Education, and the Wisconsin State Assembly from 1955 until 1973. He was a Republican."},
        ],
        [
            {"text": "Marshall Sehorn: Marshall Estus Sehorn (June 25, 1934 – December 5, 2006) was an American A&R man, songwriter, music publisher and entrepreneur who played an important role in the development of R&B and popular music in New Orleans between the 1950s and 1970s, particularly as the business partner of record producer Allen Toussaint."},
            {"text": "Toussaint (album): Toussaint is a 1971 solo funk, jazz and soul album by Allen Toussaint, his second solo album and his first since the 1950s."},
            {"text": "Alexandre de Lesseps: Alexandre de Lesseps was born in Paris, France. His education took place in Khartoum, Sudan and in France and at Northwestern University in Chicago, Illinois."},
            {"text": "Bourvil: André Bourvil, born André Robert Raimbourg (; 27 July 1917, Prétot-Vicquemare, France – 23 September 1970, Paris), often known mononymously as Bourvil, was a French actor and singer best known for his roles in comedy films, most notably in his collaboration with Louis de Funès in the films 'Le Corniaud' (1965) and 'La Grande Vadrouille' (1966). For his performance in 'Le Corniaud', he won a Special Diploma at the 4th Moscow International Film Festival."},
            {"text": "Michael Jordan: As a freshman in coach Dean Smith's team - oriented system, he was named ACC Freshman of the Year after he averaged 13.4 points per game (ppg) on 53.4% shooting (field goal percentage). He made the game - winning jump shot in the 1982 NCAA Championship game against Georgetown, which was led by future NBA rival Patrick Ewing. Jordan later described this shot as the major turning point in his basketball career. During his three seasons at North Carolina, he averaged 17.7 ppg on 54.0% shooting, and added 5.0 rebounds per game (rpg). He was selected by consensus to the NCAA All - American First Team in both his sophomore (1983) and junior (1984) seasons. After winning the Naismith and the Wooden College Player of the Year awards in 1984, Jordan left North Carolina one year before his scheduled graduation to enter the 1984 NBA draft. The Chicago Bulls selected Jordan with the third overall pick, after Hakeem Olajuwon (Houston Rockets) and Sam Bowie (Portland Trail Blazers). One of the primary reasons why Jordan was not drafted sooner was because the first two teams were in need of a center. However, Trail Blazers general manager Stu Inman contended that it was not a matter of drafting a center, but more a matter of taking Sam Bowie over Jordan, in part because Portland already had Clyde Drexler, who was a guard with similar skills to Jordan. ESPN, citing Bowie's injury - laden college career, named the Blazers' choice of Bowie as the worst draft pick in North American professional sports history. Jordan returned to North Carolina to complete his degree in 1986. He graduated the same year with a Bachelor of Arts degree in geography."},
        ],
    ]

    print("\nGenerated Prompts:\n")
    for q, hist in zip(questions, batch_history):
        prompt = answer_generator._build_prompt(q["question"], hist)
        print(f"Prompt for Q: {q['question']}\n{prompt}\n")
        print("-----------------------------")

    answers = answer_generator.batch_answer(questions, batch_history)

    print("===========================")
    print("\nGenerated Answers (Raw and Extracted):\n")
    for q, raw in zip(questions, answers):
        extracted = answer_generator.extract_answer(raw)
        print(f"Q: {q['question']}\nRaw Answer: {raw}\nExtracted Answer: {extracted}\n")
        print("-----------------------------")


if __name__ == "__main__":
    test()
