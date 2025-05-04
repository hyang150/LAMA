# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from batch_eval_KB_completion import main as run_evaluation
from batch_eval_KB_completion import load_file
from lama.modules import build_model_by_name
import pprint
import statistics
from os import listdir
import os
from os.path import isfile, join
from shutil import copyfile
from collections import defaultdict

LMs = [
    # {
    #     "lm": "transformerxl",
    #     "label": "transformerxl",
    #     "models_names": ["transformerxl"],
    #     "transformerxl_model_name": "transfo-xl-wt103",
    #     "transformerxl_model_dir": "pre-trained_language_models/transformerxl/transfo-xl-wt103/",
    # },
    # {
    #     "lm": "elmo",
    #     "label": "elmo",
    #     "models_names": ["elmo"],
    #     "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway",
    #     "elmo_vocab_name": "vocab-2016-09-10.txt",
    #     "elmo_model_dir": "pre-trained_language_models/elmo/original",
    #     "elmo_warm_up_cycles": 10,
    # },
    # {
    #     "lm": "elmo",
    #     "label": "elmo5B",
    #     "models_names": ["elmo"],
    #     "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway_5.5B",
    #     "elmo_vocab_name": "vocab-enwiki-news-500000.txt",
    #     "elmo_model_dir": "pre-trained_language_models/elmo/original5.5B/",
    #     "elmo_warm_up_cycles": 10,
    # },
    # {
    #     "lm": "bert",
    #     "label": "bert_base",
    #     "models_names": ["bert"],
    #     "bert_model_name": "bert-base-cased",
    #     "bert_model_dir": "pre-trained_language_models/bert/cased_L-12_H-768_A-12",
    # },
    # {
    #     "lm": "bert",
    #     "label": "bert_large",
    #     "models_names": ["bert"],
    #     "bert_model_name": "bert-large-cased",
    #     "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    # },
    {
        "lm": "gptj",
        "label": "gptj-6B",
        "models_names": ["gptj"],
        "gptj_model_name": "EleutherAI/gpt-j-6B",
        "batch_size": 7,
        "max_sentence_length": 1024,
        "temperature": 0.3,
        "top_p": 0.9,
        "num_return_sequences": 1,
        "max_new_tokens": 50,
        "do_sample": True,
        "evaluation_metric": "exact_match",
    }
]


def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    input_param={
        "lm": "gptj",
        "label": "gptj-6B",
        "models_names": ["gptj"],
        "gptj_model_name": "EleutherAI/gpt-j-6B",
        "batch_size": 7,
        "max_sentence_length": 1024,
        "temperature": 0.3,
        "top_p": 0.9,
        "num_return_sequences": 1,
        "max_new_tokens": 50,
        "do_sample": True,
        "evaluation_metric": "exact_match",
    },
    use_negated_probes=False,
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_scores = []
    type_scores = defaultdict(list)
    type_count = defaultdict(list)

    results_file = open("last_results.csv", "w+")
    results_file.write("relation,exact_match,f1_score\n")

    for relation in relations:
        pp.pprint(relation)
        PARAMETERS = {
            "dataset_filename": os.path.join(data_path_pre, data_path_post),
            "common_vocab_filename": "pre-trained_language_models/common_vocab_cased.txt",
            "template": relation.get("template", "Context: [X]\nQuestion: [Y]\nAnswer:"),
            "batch_size": input_param.get("batch_size", 7),
            "logdir": "output",
            "full_logdir": "output/results/{}/{}".format(
                input_param["label"], relation["relation"]
            ),
            "lowercase": False,
            "max_sentence_length": input_param.get("max_sentence_length", 1024),
            "threads": -1,
            "interactive": False,
            "use_negated_probes": use_negated_probes,
            "temperature": input_param.get("temperature", 0.3),
            "top_p": input_param.get("top_p", 0.9),
            "num_return_sequences": input_param.get("num_return_sequences", 1),
            "max_new_tokens": input_param.get("max_new_tokens", 50),
            "do_sample": input_param.get("do_sample", True),
            "evaluation_metric": input_param.get("evaluation_metric", "exact_match"),
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]
            if use_negated_probes:
                PARAMETERS["template_negated"] = relation["template_negated"]

        PARAMETERS.update(input_param)
        print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)

        # see if file exists
        try:
            data = load_file(args.dataset_filename)
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue

        if model is None:
            [model_type_name] = args.models_names
            model = build_model_by_name(model_type_name, args)

        scores = run_evaluation(args, shuffle_data=False, model=model)
        print("Scores: {}".format(scores), flush=True)
        all_scores.append(scores)

        # Write results to file
        with open("last_results.csv", "a") as f:
            if isinstance(scores, dict):
                # Handle dictionary format with exact_match and f1_score
                f.write("{},{},{}\n".format(
                    relation["relation"], 
                    scores.get("exact_match", 0.0), 
                    scores.get("f1_score", 0.0)
                ))
            else:
                # Handle float format (MRR score)
                f.write("{},{},{}\n".format(
                    relation["relation"],
                    scores,  # MRR score
                    0.0     # Default F1 score
                ))

        if "type" in relation:
            if isinstance(scores, dict):
                type_scores[relation["type"]].append(scores.get("exact_match", 0.0))
            else:
                type_scores[relation["type"]].append(scores)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(len(data))

    # Calculate mean scores
    if all_scores and isinstance(all_scores[0], dict):
        # Calculate means for each metric
        mean_scores = {
            "exact_match": statistics.mean([s.get("exact_match", 0.0) for s in all_scores]),
            "f1_score": statistics.mean([s.get("f1_score", 0.0) for s in all_scores])
        }
    else:
        mean_scores = statistics.mean(all_scores)
    
    print("@@@ {} - mean scores: {}".format(input_param["label"], mean_scores))
    results_file.close()

    for t, l in type_scores.items():
        print(
            "@@@ ",
            input_param["label"],
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )

    return mean_scores, all_scores


def get_TREx_parameters(data_path_pre="data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_parameters():
    relations = [
        {
            "relation": "place_of_birth",
            "template": "[X] was born in [Y] .",
            "template_negated": "[X] was not born in [Y] .",
        },
        {
            "relation": "date_of_birth",
            "template": "[X] (born [Y]).",
            "template_negated": "[X] (not born [Y]).",
        },
        {
            "relation": "place_of_death",
            "template": "[X] died in [Y] .",
            "template_negated": "[X] did not die in [Y] .",
        },
    ]
    data_path_pre = "data/Google_RE/"
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def get_ConceptNet_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "ConceptNet/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_Squad_parameters(data_path_pre="data/"):
    relations = [{
        "relation": "squad",
        "template": "Context: [X]\nQuestion: [Y]\nAnswer:",
        "type": "qa"
    }]
    data_path_pre += "Squad/"
    data_path_post = "test.jsonl"
    return relations, data_path_pre, data_path_post


def run_all_LMs(parameters):
    for ip in LMs:
        print(ip["label"])
        run_experiments(*parameters, input_param=ip, use_negated_probes=False)


if __name__ == "__main__":

    # print("1. Google-RE")
    # parameters = get_GoogleRE_parameters()
    # run_all_LMs(parameters)

    # print("2. T-REx")
    # parameters = get_TREx_parameters()
    # run_all_LMs(parameters)

    # print("3. ConceptNet")
    # parameters = get_ConceptNet_parameters()
    # run_all_LMs(parameters)

    print("4. SQuAD")
    parameters = get_Squad_parameters()
    run_all_LMs(parameters)

