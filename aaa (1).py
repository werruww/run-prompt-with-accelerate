from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",          # Automatically choose float32 or float16 based on device
    device_map="auto"            # Automatically map layers to available devices (e.g., GPU)
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define prompt and messages
prompt = "Who is Napoleon Bonaparte?"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# Prepare inputs for the model
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate output from the model
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

# Post-process generated output
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Print the response from the model
print(response)
