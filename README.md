# run-prompt-with-accelerate
how to run-prompt-with-accelerate without train




Qwen/Qwen2.5-7B-Instruct

mistralai/Mistral-7B-Instruct-v0.3

microsoft/phi-2

!pip install git+https://github.com/huggingface/transformers


!python3 -m pip install tensorflow[and-cuda]

# Verify the installation:

!python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
!pip install nvidia-tensorrt
!pip install -r /content/requirements-docs.txt


requirements-docs.txt
furo
myst-parser==4.0.0
sphinx<8
sphinx-copybutton
sphinx-design>=0.6.0
https://github.com/huggingface/accelerate/blob/main/examples/inference/distributed/phi2.py

!accelerate launch --num_processes 1 a.py

!accelerate launch --num_processes 1 aa.py

!accelerate launch --num_processes 1 aaa.py













colab t4

https://colab.research.google.com/drive/1E7o2HjgtTOL7-NY3Qqcw8fUnEJyfAPA7?authuser=0#scrollTo=Pd4YCFkNs6Sr


https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

https://www.tensorflow.org/install/pip

!pip install git+https://github.com/huggingface/transformers

!pip install accelerate

from accelerate import Accelerator
accelerator = Accelerator()

device = accelerator.device
model, optimizer, data = accelerator.prepare(model, optimizer, data)
accelerator.backward(loss)

import gc
gc.collect()

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import gc
gc.collect()
accelerator = Accelerator()

device = accelerator.device

model_name = "Qwen/Qwen2.5-7B-Instruct"
#model, optimizer, data = accelerator.prepare(model_name, optimizer, data)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
import gc
gc.collect()
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
import gc
gc.collect()
accelerator.backward(loss)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from accelerate import Accelerator
import torch
import gc

gc.collect()
accelerator = Accelerator()

device = accelerator.device

model_name = "Qwen/Qwen2.5-7B-Instruct"

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create an optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)  # You can adjust the learning rate

# Create a sample dataset (replace with your actual data)
data = torch.randint(0, tokenizer.vocab_size, (16, 128))

# Prepare the model, optimizer, and data with the accelerator
model, optimizer, data = accelerator.prepare(model, optimizer, data)
model = torch.nn.Transformer()
# Rest of your code...
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
import gc
gc.collect()
# Define your loss function and calculate the loss (replace with your actual loss calculation)
# Example:
# loss = some_loss_function(model_outputs, targets)

# Backpropagate the loss using the accelerator
# accelerator.backward(loss)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

!accelerate config -h

!accelerate config /content/single_gpu.yaml

!accelerate launch --config_file /content/single_gpu.yaml



from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]








!accelerate env

!git clone https://github.com/huggingface/accelerate.git

%cd /content/accelerate/examples

!pip install 'accelerate>=0.27.0' 'torchpippy>=0.2.0'



!accelerate launch --num_processes 1 /content/accelerate/examples/inference/distributed/phi2.py

!accelerate launch --num_machines=1 --mixed_precision=fp16 --dynamo_backend=no /content/accelerate/examples/inference/distributed/phi2.py

!accelerate launch --config_file default_config.yaml /content/accelerate/examples/inference/distributed/phi2.py

!accelerate launch --config_file default_config.yaml /content/accelerate/examples/inference/distributed/phi2.py

!accelerate config

!accelerate launch /content/single_gpu.yaml /content/accelerate/examples/inference/distributed/phi2.py

!pip3 install deepspeed

### شغال

!accelerate launch --num_processes 1 phi2.py

import gc
import torch

# إفراغ ذاكرة وحدة معالجة الرسومات
torch.cuda.empty_cache()

# إفراغ ذاكرة الوصول العشوائي
gc.collect()

# إفراغ ذاكرة القرص (مثال على حذف ملف)
!rm -rf /path/to/file

### شغال

!accelerate launch --num_processes 1 phi2.py







### شغال

!accelerate launch --num_processes 1 phi2.py

!huggingface-cli login

!accelerate launch --num_processes 1 phi2.py

accelerate/examples/inference/distributed
/phi2.py
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate import PartialState
from accelerate.utils import gather_object


# Start up the distributed environment without needing the Accelerator.
distributed_state = PartialState()

# You can change the model to any LLM such as mistralai/Mistral-7B-v0.1 or meta-llama/Llama-2-7b-chat-hf
model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map=distributed_state.device, torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Need to set the padding token to the eos token for generation
tokenizer.pad_token = tokenizer.eos_token

prompts = [
    "I would like to",
    "hello how are you",
    "what is going on",
    "roses are red and",
    "welcome to the hotel",
]

# You can change the batch size depending on your GPU RAM
batch_size = 2
# We set it to 8 since it is better for some hardware. More information here https://github.com/huggingface/tokenizers/issues/991
pad_to_multiple_of = 8

# Split into batches
# We will get the following results:
# [ ["I would like to", "hello how are you"], [ "what is going on", "roses are red and"], [ "welcome to the hotel"] ]
formatted_prompts = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

# Apply padding on the left since we are doing generation
padding_side_default = tokenizer.padding_side
tokenizer.padding_side = "left"
# Tokenize each batch
tokenized_prompts = [
    tokenizer(formatted_prompt, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")
    for formatted_prompt in formatted_prompts
]
# Put back the original padding behavior
tokenizer.padding_side = padding_side_default

completions_per_process = []
# We automatically split the batched data we passed to it across all the processes. We also set apply_padding=True
# so that the GPUs will have the same number of prompts, and you can then gather the results.
# For example, if we have 2 gpus, the distribution will be:
# GPU 0: ["I would like to", "hello how are you"],  "what is going on", "roses are red and"]
# GPU 1: ["welcome to the hotel"], ["welcome to the hotel"] -> this prompt is duplicated to ensure that all gpus have the same number of prompts
with distributed_state.split_between_processes(tokenized_prompts, apply_padding=True) as batched_prompts:
    for batch in batched_prompts:
        # Move the batch to the device
        batch = batch.to(distributed_state.device)
        # We generate the text, decode it and add it to the list completions_per_process
        outputs = model.generate(**batch, max_new_tokens=20)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        completions_per_process.extend(generated_text)

# We are gathering string, so we need to use gather_object.
# If you need to gather tensors, you can use gather from accelerate.utils
completions_gather = gather_object(completions_per_process)

# Drop duplicates produced by apply_padding in split_between_processes
completions = completions_gather[: len(prompts)]

distributed_state.print(completions)





Who is Napoleon Bonaparte?\n\nNapoleon Bonaparte was a French military and political leader who rose to prominence during the French Revolution and its associated wars. He was born on August 15, 1769, on the island of Corsica, which was then a possession of the Republic of Genoa. Napoleon\'s father, Carlo Buonaparte, was a lawyer and a member of the Corsican nobility, while his mother, Letizia Ramolino, was a woman of humble origins.\n\nNapoleon was educated at a Jesuit school in Ajaccio, the capital of Corsica, and later at the Royal Military Academy in Brienne-le-Château, France. He was commissioned as a second lieutenant in the French Army in 1785, and served in various capacities during the French Revolution.\n\nIn 1796, Napoleon was given command of the French Army in Italy, where he quickly established a reputation as a brilliant military strategist. He defeated the Austrian forces and expanded French control over much of Italy, earning the nickname "the Little Corporal." In 1799, he staged a coup and took control of the French government, becoming First Consul of the French Republic.\n\nAs First Consul, Napoleon consolidated his power and began a series of military campaigns that would make him one of the most powerful men in Europe. He invaded Egypt in 1798, establishing a French presence in the Middle East and challenging British control of the region. He also invaded Switzerland, Austria, and Russia, and defeated the British at the Battle of Trafalgar in 1805.\n\nIn 1804, Napoleon crowned himself Emperor of the French, and he ruled as such until his abdication in 1814. During his reign, he implemented a series of political and social reforms, including the creation of a modern legal code, the Code Napoléon, and the promotion of education and economic development.\n\nNapoleon was exiled to the island of Elba in 1814, but he escaped and returned to France in 1815. He was defeated at the Battle of Waterloo in June of that year and was exiled to the island of Saint Helena, where he died on May 5, 1821.\n\nNapoleon Bonaparte is remembered as one of the most important figures in modern European history. He is credited with transforming France and Europe through his military victories, political reforms, and cultural achievements. He is also known for his ambition, charisma, and intelligence, as well as his ruthless and authoritarian rule.']









/content/a.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


!accelerate launch --num_processes 1 a.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

!pip install tensorflow-gpu==2.13.0  # Or your desired TF version
!pip install nvidia-tensorrt

!pip install tensorflow-gpu

!nvidia-smi

pip install --upgrade pip

import tensorflow as tf
   print(tf.__version__)

شغال

!pip install git+https://github.com/huggingface/transformers

شغال

!pip install -r /content/requirements-docs.txt



شغال

!python3 -m pip install tensorflow[and-cuda]
# Verify the installation:
!python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

### شغال

!accelerate launch --num_processes 1 aa.py



Who is Napoleon Bonaparte?

!accelerate launch --num_processes 1 aaa.py

/content/phi2.py======mistralv3
/content/aaa.py====Qwen2.5-7B-Instruct


hالاكواد التى اشتغلت جيدا
/content/phi2.py
/content/aaa.py



phi2.py
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate import PartialState
from accelerate.utils import gather_object


# Start up the distributed environment without needing the Accelerator.
distributed_state = PartialState()

# You can change the model to any LLM such as mistralai/Mistral-7B-v0.1 or meta-llama/Llama-2-7b-chat-hf
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map=distributed_state.device, torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Need to set the padding token to the eos token for generation
tokenizer.pad_token = tokenizer.eos_token

prompts = [
    "Who is Napoleon Bonaparte?",
]

# You can change the batch size depending on your GPU RAM
batch_size = 2
# We set it to 8 since it is better for some hardware. More information here https://github.com/huggingface/tokenizers/issues/991
pad_to_multiple_of = 8

# Split into batches
# We will get the following results:
# [ ["I would like to", "hello how are you"], [ "what is going on", "roses are red and"], [ "welcome to the hotel"] ]
formatted_prompts = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

# Apply padding on the left since we are doing generation
padding_side_default = tokenizer.padding_side
tokenizer.padding_side = "left"
# Tokenize each batch
tokenized_prompts = [
    tokenizer(formatted_prompt, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")
    for formatted_prompt in formatted_prompts
]
# Put back the original padding behavior
tokenizer.padding_side = padding_side_default

completions_per_process = []
# We automatically split the batched data we passed to it across all the processes. We also set apply_padding=True
# so that the GPUs will have the same number of prompts, and you can then gather the results.
# For example, if we have 2 gpus, the distribution will be:
# GPU 0: ["I would like to", "hello how are you"],  "what is going on", "roses are red and"]
# GPU 1: ["welcome to the hotel"], ["welcome to the hotel"] -> this prompt is duplicated to ensure that all gpus have the same number of prompts
with distributed_state.split_between_processes(tokenized_prompts, apply_padding=True) as batched_prompts:
    for batch in batched_prompts:
        # Move the batch to the device
        batch = batch.to(distributed_state.device)
        # We generate the text, decode it and add it to the list completions_per_process
        outputs = model.generate(**batch, max_new_tokens=1024)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        completions_per_process.extend(generated_text)

# We are gathering string, so we need to use gather_object.
# If you need to gather tensors, you can use gather from accelerate.utils
completions_gather = gather_object(completions_per_process)

# Drop duplicates produced by apply_padding in split_between_processes
completions = completions_gather[: len(prompts)]

distributed_state.print(completions)


/content/aaa.py
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


/content/aa.py
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
prompt = "Give me a short introduction to large language models."
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


/content/requirements-docs.txt
furo
myst-parser==4.0.0
sphinx<8
sphinx-copybutton
sphinx-design>=0.6.0


https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
