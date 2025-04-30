#!/usr/bin/env python
# File: run_experiments_gpt_j6.py

import json
import os
import torch
from lama.modules.gptj_connector import GPTJ
from lama.evaluation_metrics import get_ranking

def load_squad_samples(path):
    """
    读取 SQuAD JSONL，每行包含 {context, question, answers},
    返回 [{prompt: "... [MASK].", label: answer_text}, ...]
    """
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            ctx = obj.get('context', '').strip()
            q   = obj.get('question', '').strip()
            prompt = f"{ctx} Question: {q} Answer: [MASK]."

            # 提取第一个答案文本
            answers = obj.get('answers', [])
            if isinstance(answers, list) and answers:
                first = answers[0]
                if isinstance(first, dict) and 'text' in first:
                    label = first['text']
                else:
                    label = str(first)
            else:
                label = ""
            samples.append({'prompt': prompt, 'label': label})
    return samples

def main():
    # —— 1. 加载模型 —— 
    MODEL_NAME = "EleutherAI/gpt-j-6B"
    print("Loading GPT-J model…")
    class Args: pass
    args = Args()
    args.gptj_model_name = MODEL_NAME
    # 以下两个属性在 GPTJ 初始化时不会用到，但接口需要：
    args.common_vocab_filename = None
    args.use_negated_probes = False

    connector = GPTJ(args)
    connector.model.eval()

    # —— 2. 载入所有 SQuAD 样本 —— 
    data_path = os.path.join("data", "Squad", "test.jsonl")
    samples = load_squad_samples(data_path)
    n = len(samples)
    print(f"{n} samples loaded from {data_path}\n")

    # —— 3. 遍历评估 —— 
    p1_list = []
    for i, sample in enumerate(samples):
        prompt = sample['prompt']
        label  = sample['label']

        # 获取 log-probs 和 mask 位置
        # GPTJ.get_batch_generation 需要传入 List[List[str]]
        log_probs_list, token_ids_list, masked_indices_list = connector.get_batch_generation([[prompt]])
        log_probs      = log_probs_list[0]       # Tensor [seq_len, vocab_size]
        masked_indices = masked_indices_list[0]   # List[int]

        # 将 label 转为 token ID（取第一个），若失败则设为 None
        ids = connector.get_id(label)
        label_index = ids[0] if (isinstance(ids, list) and ids) else None

        # 计算 MRR, P@X, top-k list
        # 这里我们只关心 P@1，topk=10 用于可视化
        MRR, P_at_1, result_dict, topk_msg = get_ranking(
            log_probs,
            masked_indices,
            connector.vocab,
            label_index=label_index,
            index_list=None,
            topk=10,
            P_AT=1,
            print_generation=False
        )
        p1_list.append(P_at_1)

        # 打印前 5 条的 prompt 和 Top-10 候选
        if i < 5:
            print(f"—— Sample {i+1}/{n} ——")
            print(prompt)
            print(topk_msg)
            print("\n")

        # 每隔 50 条打印一次累积 Precision@1
        if (i+1) % 50 == 0:
            avg_p1 = sum(p1_list) / len(p1_list)
            print(f"[{i+1}/{n}] so far Precision@1 = {avg_p1:.4f}")

    # —— 4. 输出整体结果 —— 
    overall_p1 = sum(p1_list) / n
    print(f"\n==== Final Precision@1 over {n} samples: {overall_p1:.4f} ====")

if __name__ == "__main__":
    main()
