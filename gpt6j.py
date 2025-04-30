#!/usr/bin/env python
# File: scripts/run_experiments_gpt_j6.py

import json, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_squad(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            ctx = obj.get('context','').strip()
            q   = obj.get('question','').strip()
            # 拼接 Prompt
            prompt = f"Passage: {ctx}\nQuestion: {q}\nAnswer:"
            # 取第一个答案作为标签（可选，用于后续打分）
            answers = obj.get('answers',[])
            if isinstance(answers, list) and answers:
                a0 = answers[0]
                if isinstance(a0, dict) and 'text' in a0:
                    label = a0['text']
                else:
                    label = str(a0)
            else:
                label = ""
            out.append({'prompt':prompt, 'label':label})
    return out

def main():
    MODEL = "EleutherAI/gpt-j-6B"
    print("Loading GPT-J…")
    tok = AutoTokenizer.from_pretrained(MODEL)
    m   = AutoModelForCausalLM.from_pretrained(
              MODEL, torch_dtype=torch.float16, device_map="auto"
          )
    m.eval()

    data_path = os.path.join("data","Squad","test.jsonl")
    samples = load_squad(data_path)
    print(f"{len(samples)} samples loaded.\n")

    # 打开文件收集所有输出
    fout = open("gptj_generation_results.txt","w",encoding="utf-8")

    for i, s in enumerate(samples, 1):
        prompt, label = s['prompt'], s['label']
        inputs = tok(prompt, return_tensors="pt").to(m.device)
        out = m.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            eos_token_id=tok.eos_token_id
        )
        gen = tok.decode(out[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        # 写入文件
        fout.write(f"=== Sample {i} ===\n")
        fout.write(f"Prompt:\n{prompt}\n")
        fout.write(f"GPT-J Answer: {gen}\n")
        fout.write(f"Label (gold): {label}\n\n")

        # 可选：每 50 条打印进度
        if i % 50 == 0:
            print(f"[{i}/{len(samples)}] done.")

    fout.close()
    print("\nAll done. See gptj_generation_results.txt")

if __name__=="__main__":
    main()
