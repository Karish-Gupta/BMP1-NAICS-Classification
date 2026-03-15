import torch
from huggingface_hub import login
from unsloth import FastLanguageModel
import os
import re

# Login to HF CLI
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])

class QuestionGenerationModel:
    def __init__(self, model_name="Qwen/Qwen3-4B-Thinking-2507"):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = 8192, # Increased for Qwen3-Thinking depth
            dtype = None,
            load_in_4bit = True,
        )
        
        FastLanguageModel.for_inference(self.model)
        self.model.eval()
        print(f"Model {model_name} loaded successfully.")
        
    def invoke(self, prompt):
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
        # Use skip_special_tokens=False if you want to see the <think> tags for debugging
        generated_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)

        return generated_response
        
    def generate_questions(self, structured_output):
        system_prompt = """
        You are an AI assistant aiding law enforcement analysts reviewing body-worn camera footage.

        Your task:
        - Based on the provided structured details, generate a list of investigative questions.
        - Every question must be something a human could answer by watching the video.
        - The goal is to guide analysts toward visual clues, context, behavior, or environment details that may matter.

        Rules for your output:
        - Write a total of 4 meaningful questions that can extract the most visual information.
        - Do NOT repeat facts already stated.
        - Focus areas include: body language, environment, timeline, objects, threat indicators, interaction dynamics, or visual anomalies.
        - Use clear, concise, professional language.

        FORMATTING RULES:
        1. You must start with a hidden reasoning block using <think>...</think> tags.
        2. Inside the <think> block, analyze the scene, entity, and actions.
        3. After reasoning, provide the final output inside <question>...</question> tags.
        4. The content inside <question> tags must be a numbered list of 4 questions. 
        """
        
        user_prompt = f"Structured information provided:\n {structured_output}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        raw_response = self.invoke(prompt)
        
        # Isolate the actual response from the thought process
        if "</think>" in raw_response:
            # We take everything after the first occurrence of </think>
            actual_output = raw_response.split("</think>", 1)[1].strip()
        else:
            actual_output = raw_response.strip()

        # Try to extract from <question> tags within that clean output
        match = re.search(r"<question>(.*?)</question>", actual_output, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Fallback - return the clean output without the "think" block
        print("Warning: <question> tags not found in final response.")
        return actual_output