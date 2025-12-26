from ctransformers import AutoModelForCausalLM
from pathlib import Path

MODEL_PATH = r"D:\AI_Class\models\Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

print("=" * 80)
print("🔍 Testing LLaMA 3 GGUF Model (CPU)")
print("=" * 80)

model_file = Path(MODEL_PATH)

if not model_file.exists():
    raise FileNotFoundError("Model file not found")

print(f"Model: {model_file.name}")
print(f"Size : {model_file.stat().st_size / (1024**3):.2f} GB\n")

print("⏳ Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    model_type="llama",     # ✅ CORRECT
    max_new_tokens=128,
    context_length=4096,
    temperature=0.2,
    threads=8               # adjust to CPU cores
)

print("✅ Model loaded successfully\n")

prompt = "Explain IPC Section 420 in simple terms."

print("🧪 Inference test:\n")
print(model(prompt))
