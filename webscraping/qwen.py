from unsloth import FastLanguageModel
from huggingface_hub import login
import torch
import os

# Login to HF CLI
if "HF_TOKEN" in os.environ:
   login(token=os.environ["HF_TOKEN"])

class QwenLLM:
   def __init__(self, model_name="Qwen/Qwen3-14B"):
      self.model, self.tokenizer = FastLanguageModel.from_pretrained(
         model_name = model_name,
         max_seq_length = 2048,
         dtype = None,
         load_in_4bit = True,
      )
      
      FastLanguageModel.for_inference(self.model)
      self.model.eval()
      print(f"Model {model_name} loaded successfully.")
      
   def generate(self, prompt):
      inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
   
      # Generate
      with torch.no_grad():
         outputs = self.model.generate(
               **inputs, 
               max_new_tokens=2048, 
               temperature=0.1,
               top_p=0.95,
               do_sample=True,
               pad_token_id=self.tokenizer.pad_token_id
         )

      generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
      generated_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)

      return generated_response
      
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

      prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      
      raw_response = self.generate(prompt)
      
      # Isolate the actual response from the thought process
      if "</think>" in raw_response:
         # We take everything after the first occurrence of </think>
         actual_output = raw_response.split("</think>", 1)[1].strip()
      else:
         actual_output = raw_response.strip()

      actual_output = actual_output.replace("<|im_end|>", "").strip() # Remove any generation prompt artifacts
      
      return actual_output