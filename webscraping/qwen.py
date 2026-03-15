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
      system_prompt = """
      You are a webscraping assistant designed to summarize information from unstructured webscraped data and generate a clear, concise summarizations about the given text.

      Rules for your output:
      1. Keep summarization concise and focused on key insights.
      2. Do not include any information that is not directly from the provided information.
      3. If the provided information is insufficient to generate a meaningful summary, respond with "N/A".

      SUMMARY MUST BE 1 or 2 SENTENCES, MAX 20 WORDS, AND FOCUS ON THE MOST IMPORTANT ASPECTS OF THE PROVIDED INFORMATION.
      """
      
      user_prompt = f"Unstructured information provided:\n {unstructured_text}"

      messages = [
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}
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