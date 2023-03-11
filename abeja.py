from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
model = AutoModelForCausalLM.from_pretrained("abeja/gpt-neox-japanese-2.7b")

input_text = "Hagging Face とは何ですか？"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
gen_tokens = model.generate(
    input_ids,
    max_length=100,
    do_sample=True,
    num_return_sequences=3,
    top_p=0.95,
    top_k=50,
)

for gen_text in tokenizer.batch_decode(gen_tokens, skip_special_tokens=True):
    print(gen_text)
