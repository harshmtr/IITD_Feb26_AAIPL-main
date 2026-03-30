import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class AAgent:

    def __init__(self, model_name="models--Unsloth--Llama-3.1-8B-", device="cuda"):

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map={"": 0},  # force full GPU
        )

        self.model.eval()

    def generate_batch(self, prompts, max_new_tokens=96):

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768  # faster attention
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,      # greedy = fastest
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        responses = []

        for inp, gen in zip(inputs.input_ids, outputs):
            output_ids = gen[len(inp):]
            responses.append(
                self.tokenizer.decode(output_ids, skip_special_tokens=True)
            )

        return responses
