from ctransformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    r"D:\AI_Class\models\Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    model_type="llama"
)

output = model("Hello world!", max_new_tokens=50)
print(output)
