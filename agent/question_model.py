import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QAgent(object):

    def __init__(self, **kwargs):

        model_name = "/root/.cache/huggingface/hub/hf_models/models--Unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True
        )

        # ✅ CRITICAL FIX
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,      # use fp16 (safer on ROCm)
            device_map="auto",              # allow split across GPU/CPU
            low_cpu_mem_usage=True,
            offload_folder="offload",       # enable disk offload
            offload_state_dict=True,
            trust_remote_code=True
        )

        self.model.eval()

    def generate_response(self, message, system_prompt=None, **kwargs):

        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        if isinstance(message, str):
            message = [message]

        texts = []
        for msg in message:
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]

            text = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512   # reduce memory
        ).to(self.model.device)

        start_time = time.time()

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id
        )

        generation_time = time.time() - start_time

        outputs = []
        token_len = 0

        for inp, gen in zip(inputs.input_ids, generated_ids):
            output_ids = gen[len(inp):]
            token_len += len(output_ids)

            outputs.append(
                self.tokenizer.decode(output_ids, skip_special_tokens=True)
            )

        return (
            outputs if len(outputs) > 1 else outputs[0],
            token_len,
            generation_time,
        )
