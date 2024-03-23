import os
import torch
from typing import List, Mapping, Any, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
from med_assist.config import CONFIG

class Llama2(LLM):
    hf_token: Optional[str] = None
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None
    adapter_path: Optional[str] = None

    def __init__(self, hf_token=None, tokenizer=None, model=None, adapter_path=None) -> None:
        super().__init__()

        if self.hf_token == None:
            self.hf_token = os.environ['HUGGINGFACE_API_KEY']

        if self.tokenizer == None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=CONFIG["llama"]["path"],
                token=self.hf_token
                )

            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.model == None:

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
                )

            model_config = AutoConfig.from_pretrained(
                CONFIG['llama']['path'],
                device_map="auto",
                do_sample=True,
                temperature=0.25,
                torch_dtype=torch.bfloat16,
                max_new_tokens=512,
                max_length=4096,
                num_return_sequences=1,
                top_p=1
                )
            model_config.pad_token_id = model_config.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=CONFIG["llama"]["path"],
                config=model_config,
                quantization_config=quantization_config,
                token=self.hf_token
            )

        if self.adapter_path != None:
            
            self.model = PeftModel. \
                from_pretrained(self.model, self.adapter_path). \
                merge_and_unload()

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        max_length = self.model.config.max_length
        
        input_tokens = self.tokenizer(prompt, return_tensors="pt", padding=True).input_ids[:, :max_length].to("cuda")
        output_tokens = self.model.generate(input_tokens)
        generated_tokens = output_tokens[:, input_tokens.shape[1]:]
        result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        return result

    @property
    def _llm_type(self) -> str:
        return "llama2"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}