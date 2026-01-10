# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("qwen2.5-1.5B")
model = AutoModelForCausalLM.from_pretrained("qwen2.5-1.5B")
print(tokenizer.is_fast)
print(model)
messages = [
    {"role": "user", "content": "Who are you?"},
]
# inputs = tokenizer.apply_chat_template(
# 	messages,
# 	add_generation_prompt=True,
# 	tokenize=True,
# 	return_dict=True,
# 	return_tensors="pt",
# ).to(model.device)
#
# outputs = model.generate(**inputs, max_new_tokens=40)
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

datapath=f"../dataset/mobvoi_seq_monkey_general_open_corpus.jsonl"
ds=load_dataset("json", data_files=datapath,split="train[:1000]")
print(ds[0])