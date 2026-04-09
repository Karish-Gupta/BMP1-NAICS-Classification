from vllm import LLM, SamplingParams
from huggingface_hub import login
import os

# Login to HF CLI
if "HF_TOKEN" in os.environ:
   login(token=os.environ["HF_TOKEN"])

class QwenLLM:
   def __init__(self, model_name="Qwen/Qwen3-14B"):
      # Initialize vLLM with 4-bit quantization to replace unsloth's load_in_4bit
      self.llm = LLM(
         model=model_name,
         quantization="bitsandbytes",
         load_format="bitsandbytes",
         max_model_len=4096, # Ensures enough context for input (2048) + output (2048)
         trust_remote_code=True,
         enforce_eager=True # Recommended for local Windows setups to prevent CUDA graph crashes
      )
      
      # Generation parameters
      self.sampling_params = SamplingParams(
         temperature=0.1,
         top_p=0.95,
         max_tokens=2048,
      )
      print(f"Model {model_name} loaded successfully via vLLM.")
         
   def invoke(self, unstructured_text):
      prompt = f"""
      You are an expert in NAICS business classification.
      Analyze the scraped text to extract a concise business profile.   
      
      Rules for your output:
      - Focus strictly on: What do they make/do? Who is their customer? How do they do it?
      - 1 or 2 sentences
      - Use only provided information
      - If not about a business OR unclear → return exactly: 'Insufficient content for summary.'
      
      Scraped Business Text:
      {unstructured_text}
      """
      
      messages = [
         {"role": "user", "content": prompt}
      ]
      
      # vLLM's chat API automatically applies the tokenizer's chat template
      outputs = self.llm.chat(
         messages=messages,
         sampling_params=self.sampling_params,
         use_tqdm=False 
      )
      
      raw_response = outputs[0].outputs[0].text
      
      # Isolate the actual response from the thought process
      if "</think>" in raw_response:
         actual_output = raw_response.split("</think>", 1)[1].strip()
      else:
         actual_output = raw_response.strip()

      return actual_output.replace("<|im_end|>", "").strip()