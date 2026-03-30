#!/usr/bin/python3

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any
import random
import json

# IMPORTANT: use normal model
from .question_model import QAgent
# from .question_model_llama import QAgent


class QuestioningAgent(object):
    """Agent responsible for generating MCQ questions."""

    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)

    # ----------------------------------------------------
    # Prompt Builder
    # ----------------------------------------------------
    def build_prompt(self, topic: str) -> Tuple[str, str]:

        system_prompt = (
            "You are an expert reasoning examiner. "
            "Generate only valid JSON and no extra text."
        )

        correct_option = random.choice(["A", "B", "C", "D"])
        wrong_opts = ", ".join([x for x in "ABCD" if x != correct_option])

        prompt = f"""
Create ONE challenging puzzle-based MCQ.

TOPIC:
{topic}

RULES:
- Exactly 4 options (A, B, C, D)
- Only option {correct_option} must be correct
- Options {wrong_opts} must be plausible but incorrect
- Explanation must be under 100 words
- Output STRICTLY valid JSON

FORMAT:
{{
  "topic": "{topic}",
  "question": "...",
  "choices": [
    "A) ...",
    "B) ...",
    "C) ...",
    "D) ..."
  ],
  "answer": "{correct_option}",
  "explanation": "..."
}}
"""
        return prompt, system_prompt

    # ----------------------------------------------------
    def generate_question(self, topic, **gen_kwargs):

        if isinstance(topic, list):
            prompts = []
            for t in topic:
                p, sp = self.build_prompt(f"{t[0]}/{t[1]}")
                prompts.append(p)
        else:
            prompts, sp = self.build_prompt(f"{topic[0]}/{topic[1]}")

        return self.agent.generate_response(prompts, sp, **gen_kwargs)

    # ----------------------------------------------------
    def populate_topics(self, topics, num_questions):
        all_subtopics = [(t, st) for t, subs in topics.items() for st in subs]
        return random.choices(all_subtopics, k=num_questions)

    # ----------------------------------------------------
    def generate_batches(
        self,
        num_questions: int,
        topics: Dict[str, List[str]],
        batch_size: int = 5,
        **kwargs,
    ):

        topic_list = self.populate_topics(topics, num_questions)

        questions, tls, gts = [], [], []

        total_batches = (len(topic_list) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS:")

        for i in range(0, len(topic_list), batch_size):
            batch_topics = topic_list[i:i + batch_size]

            q, tl, gt = self.generate_question(batch_topics, **kwargs)

            if isinstance(q, str):
                q = [q]

            questions.extend(q)
            tls.append(tl)
            gts.append(gt)

            pbar.update(1)

        pbar.close()
        return questions, tls, gts

    # ----------------------------------------------------
    def filter_questions(self, questions):

        filtered = []

        for q in questions:
            try:
                obj = json.loads(q) if isinstance(q, str) else q

                if (
                    isinstance(obj, dict)
                    and "topic" in obj
                    and "question" in obj
                    and "choices" in obj
                    and "answer" in obj
                    and isinstance(obj["choices"], list)
                    and len(obj["choices"]) == 4
                    and obj["answer"] in ["A", "B", "C", "D"]
                ):
                    filtered.append(obj)

            except Exception:
                continue

        return filtered

    # ----------------------------------------------------
    def save_questions(self, questions, file_path):

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(questions, f, indent=4)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=20)
    parser.add_argument("--output_file", type=str, default="outputs/questions.json")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load topics
    with open("assets/topics.json") as f:
        topics = json.load(f)

    agent = QuestioningAgent()

    gen_kwargs = {"tgps_show": True}
    with open("qgen.yaml") as f:
        gen_kwargs.update(yaml.safe_load(f))

    questions, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
        **gen_kwargs
    )

    if args.verbose:
        print(f"Generated {len(questions)} questions.")

    agent.save_questions(questions, args.output_file)

    filtered = agent.filter_questions(questions)
    agent.save_questions(
        filtered,
        args.output_file.replace("questions.json", "filtered_questions.json"),
    )

    print(f"Saved to {args.output_file}")