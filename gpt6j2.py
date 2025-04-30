#!/usr/bin/env python
# File: scripts/run_squad_qa.py

import json, os
from transformers import pipeline

def load_squad(path):
    out = []
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            ctx = obj.get('context','').strip()
            q   = obj.get('question','').strip()
            out.append({'context':ctx,'question':q,'answers':obj.get('answers',[])})
    return out

def main():
    # 初始化 HuggingFace SQuAD 流水线
    qa = pipeline(
        "question-answering",
        model="distilbert-base-uncased-distilled-squad",
        tokenizer="distilbert-base-uncased-distilled-squad",
        device=0  # 如果有 GPU，或去掉这行用 CPU
    )

    data = load_squad(os.path.join("data","Squad","test.jsonl"))
    fout = open("extractive_qa_results.txt","w",encoding="utf-8")

    for i, sample in enumerate(data,1):
        res = qa(question=sample['question'], context=sample['context'])
        allen = res['answer']
        score = res['score']
        fout.write(f"=== Sample {i} ===\n")
        fout.write(f"Question: {sample['question']}\n")
        fout.write(f"Predicted Answer: {allen}\n")
        fout.write(f"Confidence: {score:.4f}\n")
        # 如果想对比 gold answer:
        golds = sample.get('answers',[])
        first = golds[0]['text'] if golds and isinstance(golds[0],dict) else (golds[0] if golds else "")
        fout.write(f"Gold Answer: {first}\n\n")
        if i % 50 == 0:
            print(f"[{i}/{len(data)}] done.")

    fout.close()
    print("\nAll done. See extractive_qa_results.txt")

if __name__=="__main__":
    main()
