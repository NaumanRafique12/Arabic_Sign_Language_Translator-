# src/llm_grammar.py
import os
from llama_cpp import Llama
import threading
import time

FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, "Qwen2.5-1.5B-Instruct-GGUF", FILENAME)

# Global lock
model_lock = threading.Lock()
llm = None

def load_llm():
    global llm
    if llm is None:
        try:
            # Lower context size for speed
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=256,        
                n_threads=6,
                n_gpu_layers=-1, ## If you are using CPU pls remove this Parameter
                verbose=False
            )
            print("✅ LLM Loaded (Strict Mode)")
        except Exception as e:
            print(f"❌ LLM Load Error: {e}")

def format_arabic_sentence(words, gender):
    if not words or llm is None: 
        return " ".join(words)
    
    words_str = ' '.join(words)
    gender_rule = "female" if gender == "Female" else "male"
    
    prompt = f"""<|im_start|>system
Task: Fix the grammar of the input words.
Rules:
1. Use ONLY the provided words.
2. Conjugate verbs for {gender_rule}.
3. You may add necessary prepositions (like في, إلى, بـ) for grammar.
4. DO NOT add any new nouns, adjectives, extra context or new information .
5. Output ONLY the Arabic sentence.<|im_end|>
<|im_start|>user
{words_str}<|im_end|>
<|im_start|>assistant
"""

    try:
        with model_lock:
            output = llm(
                prompt,
                max_tokens=40,
                stop=["<|im_end|>", "\n", "Input:"],
                temperature=0.1,   # Near zero = Deterministic (No randomness)
                top_p=0.2,
                echo=False
            )
        
        result = output['choices'][0]['text'].strip()
        
        # Safety check: If result is empty or too short, return raw words
        if len(result) < 2:
            return words_str
        return result
            
    except Exception as e:
        print(f"⚠️ LLM Error: {e}")
        return words_str

# Load immediately on import
load_llm()