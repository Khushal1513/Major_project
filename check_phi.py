from ctransformers import AutoModelForCausalLM
import traceback
MODEL=r"D:\AI_Class\models\models--microsoft--Phi-3-mini-4k-instruct-gguf\snapshots\a64113399c2f6b8ad3e11c394733a2ddadaa7f33\Phi-3-mini-4k-instruct-q4.gguf"
try:
    AutoModelForCausalLM.from_pretrained(MODEL)
    print("LOAD SUCCESS")
except Exception:
    traceback.print_exc()