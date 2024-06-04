# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
from_language = 'English'
to_language = 'German'
query = 'The house is wonderful.'
target = f"translate {from_language} to {to_language}: {query}"
input_ids = tokenizer(target, return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
