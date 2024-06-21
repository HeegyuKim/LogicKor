from transformers import pipeline
from datasets import load_dataset


gen = pipeline("text-generation", "heegyu/0608-Qwen2-7B-sft-en2ko", revision="lr2e-5-epoch-2", torch_dtype="auto", device_map="cuda:3")
ds = load_dataset("GAIR/lima", split="train")

for item in ds.select(range(10)):
    prompt = [
        {"role": "system", "content": 'You are a Korean translator. Translate your English text into Korean'},
        {"role": "user", "content": item['conversations'][0]}
        ]
    print(gen.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prefix=True))

    # output = gen(prompt, max_new_tokens=1024, do_sample=False)[0]['generated_text'][-1]['content']
    # print(item['conversations'][0])
    # print("--->", output)
    # print("==" * 20)