import json
from tqdm import tqdm
from agents.answer_model import AAgent


class AnsweringAgent:

    def __init__(self):
        self.agent = AAgent()

    def build_prompt(self, question_data):

        return f"""
You are solving a multiple choice question.

Question:
{question_data["question"]}

Choices:
{question_data["choices"][0]}
{question_data["choices"][1]}
{question_data["choices"][2]}
{question_data["choices"][3]}

Return STRICT JSON only:

{{
  "answer": "A",
  "reasoning": "brief reasoning under 100 words"
}}
"""

    def answer_batches(self, questions, batch_size=8):

        all_answers = []

        for i in tqdm(range(0, len(questions), batch_size), desc="STEPS"):
            batch = questions[i:i + batch_size]

            prompts = [self.build_prompt(q) for q in batch]

            raw_outputs = self.agent.generate_batch(prompts)

            for out in raw_outputs:
                try:
                    start = out.find("{")
                    end = out.rfind("}") + 1
                    parsed = json.loads(out[start:end])
                    all_answers.append(parsed)
                except:
                    # fallback safe answer
                    all_answers.append({
                        "answer": "A",
                        "reasoning": "Fallback answer due to parsing error."
                    })

        return all_answers


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        questions = json.load(f)

    agent = AnsweringAgent()

    answers = agent.answer_batches(
        questions,
        batch_size=8
    )

    with open(args.output_file, "w") as f:
        json.dump(answers, f, indent=4)

    print(f"\nGenerated {len(answers)} answers.")
    print(f"Saved to {args.output_file}")
