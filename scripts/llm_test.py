import os
from dotenv import load_dotenv
load_dotenv()

# Ensure project root is on sys.path so imports like `groq_client` work
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

"""
Preferred behavior: try a local GGUF model first; if loading or inference fails,
fall back to remote Groq inference when `GROQ_API_KEY` is available.
"""

prompt = "Explain Section 302 of IPC in simple words."

# Try local model first
try:
    from ctransformers import AutoModelForCausalLM
    model_path = r"D:\AI_Class\models\Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    print("Trying local model at:", model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, model_type="llama3", gpu_layers=0)
    print("✅ Local model loaded successfully — using local inference")
    output = model(prompt)
    print("\n📝 Output:\n", output)
except Exception as local_err:
    print("⚠️  Local model failed:", local_err)

    # If GROQ_API_KEY is configured, attempt remote Groq inference as fallback
    if os.getenv("GROQ_API_KEY"):
        try:
            from groq_client import generate as groq_generate
            print("✅ GROQ_API_KEY detected — falling back to Groq remote inference")
            out = groq_generate(prompt, max_tokens=256, temperature=0.2, model=os.getenv("GROQ_MODEL"))
            print("\n📝 Groq Output:\n", out)
        except Exception as groq_err:
            print("❌ Groq inference also failed:", groq_err)
    else:
        print("ℹ️  GROQ_API_KEY not set — cannot fall back to remote inference. Set GROQ_API_KEY in your environment or .env to enable.")
