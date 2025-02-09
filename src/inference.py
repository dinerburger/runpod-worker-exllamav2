import os
from exllamav2 import (
  ExLlamaV2Tokenizer, 
  ExLlamaV2, 
  ExLlamaV2Config, 
  ExLlamaV2Lora, 
  ExLlamaV2Cache, 
  ExLlamaV2Cache_8bit, 
  ExLlamaV2Cache_Q8, 
  ExLlamaV2Cache_Q4, 
  ExLlamaV2Cache_Q6
)
from exllamav2.generator import (
    ExLlamaV2Sampler,
    ExLlamaV2StreamingGenerator,
)
import time
import logging
import json

MODEL_NAME = os.environ.get("MODEL_NAME")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
LORA_NAME = os.environ.get("LORA_ADAPTER_NAME", None)
LORA_REVISION = os.environ.get("LORA_ADAPTER_REVISION", "main")
MODEL_BASE_PATH = os.environ.get("MODEL_BASE_PATH", "/runpod-volume/")

def get_local_args():
    """
    Retrieve local arguments from a JSON file.

    Returns:
        dict: Local arguments.
    """
    if not os.path.exists("./local_model_args.json"):
        return {}

    with open("./local_model_args.json", "r") as f:
        local_args = json.load(f)

    if local_args.get("MODEL_NAME") is None:
        raise ValueError("Model name not found in /local_model_args.json. There was a problem when baking the model in.")

    logging.info(f"Using baked in model with args: {local_args}")
    return local_args

class Predictor:
    def setup(self):
        args = get_local_args()
        model_name = args.get("MODEL_NAME")
        model_directory = f"{MODEL_BASE_PATH}/{model_name}"

        config = ExLlamaV2Config()
        config.model_dir = model_directory
        config.prepare()

        self.tokenizer = ExLlamaV2Tokenizer(config)
        self.model = ExLlamaV2(config)
        self.model.load()
        
        cache_type: str = args.get("KV_CACHE_QUANT", "FP16").upper()
        if cache_type == "Q4":
            self.cache = ExLlamaV2Cache_Q4(self.model)
        elif cache_type == "Q6":
            self.cache = ExLlamaV2Cache_Q6(self.model)
        elif cache_type == "Q8":
            self.cache = ExLlamaV2Cache_Q8(self.model)
        elif cache_type == "FP8":
            self.cache = ExLlamaV2Cache_8bit(self.model)
        else:
            self.cache = ExLlamaV2Cache(self.model)
        

        self.settings = ExLlamaV2Sampler.Settings()

        # Load LORA adapter if specified
        self.lora_adapter = None
        if LORA_NAME is not None:
            lora_directory = f"{MODEL_BASE_PATH}{LORA_NAME.split('/')[1]}"
            self.lora_adapter = ExLlamaV2Lora.from_directory(self.model, lora_directory)

    def predict(self, settings):
        ### Set the generation settings
        self.settings.temperature = settings["temperature"]
        self.settings.smoothing_factor = settings["smoothing_factor"]
        self.settings.min_temp = settings["min_temp"]
        self.settings.max_temp = settings["max_temp"]
        self.settings.temp_exponent = settings["temp_exponent"]

        self.settings.top_p = settings["top_p"]
        self.settings.top_k = settings["top_k"]
        self.settings.top_a = settings["top_a"]
        self.settings.min_p = settings["min_p"]

        self.settings.token_repetition_penalty = settings["token_repetition_penalty"]
        self.settings.token_repetition_range = settings["token_repetition_range"]
        self.settings.token_repetition_decay = settings["token_repetition_decay"]

        self.settings.token_frequency_penalty = settings["token_frequency_penalty"]
        self.settings.token_presence_penalty = settings["token_presence_penalty"]

        output = None
        time_begin = time.time()
        output = self.streamGenerate(settings["prompt"], settings["max_new_tokens"])
        for chunk in output:
            yield chunk
        time_end = time.time()
        print(f"⏱️ Time taken for inference: {time_end - time_begin} seconds")

    def streamGenerate(self, prompt, max_new_tokens):
        input_ids = self.tokenizer.encode(prompt)
        generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
        generator.warmup()

        generator.begin_stream(input_ids, self.settings, loras=self.lora_adapter)
        generated_tokens = 0

        while True:
            chunk, eos, _ = generator.stream()
            generated_tokens += 1
            yield chunk

            if eos or generated_tokens == max_new_tokens or chunk == "</s>":
                break
