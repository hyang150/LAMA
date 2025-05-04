# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from lama.modules import build_model_by_name
import lama.utils as utils
from lama.utils import print_sentence_predictions, load_vocab
import lama.options as options
from tqdm import tqdm
from random import shuffle
import os
import json
import spacy
import lama.modules.base_connector as base
from pprint import pprint
import logging.config
import logging
import pickle
from multiprocessing.pool import ThreadPool
import multiprocessing
import lama.evaluation_metrics as metrics
import time, sys
import torch


def load_file(filename):
    data = []
    with open(filename, "r", encoding='utf-8') as f:
        if filename.endswith('.jsonl'):
            for line in f:
                try:
                    sample = json.loads(line)
                    # Check if this is a SQuAD format sample
                    if 'masked_sentences' in sample and 'obj_label' in sample:
                        # Get the masked sentence and answer
                        masked_sentence = sample['masked_sentences'][0]
                        answer = sample['obj_label']
                        
                        # Create a new masked sentence with proper format
                        new_masked_sentence = f"Context: {masked_sentence}\nQuestion: What is the answer?\nAnswer: [MASK]"
                        
                        data.append({
                            "sub_label": sample.get('sub_label', 'Squad'),
                            "obj_label": answer,
                            "masked_sentences": [new_masked_sentence],
                            "uuid": sample.get('id', str(len(data)))
                        })
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")
                    continue
        else:
            for line in f.readlines():
                data.append(json.loads(line))
    return data


def create_logdir_with_timestamp(base_logdir, modelname):
    timestr = time.strftime("%Y%m%d_%H%M%S")

    # create new directory
    log_directory = "{}/{}_{}/".format(base_logdir, modelname, timestr)
    os.makedirs(log_directory)

    path = "{}/last".format(base_logdir)
    try:
        os.unlink(path)
    except Exception:
        pass
    os.symlink(log_directory, path)
    return log_directory


def parse_template(template, subject_label, object_label):
    """Parse the template and replace placeholders with actual values."""
    SUBJ_SYMBOL = "[X]"
    OBJ_SYMBOL = "[Y]"
    template = template.replace(SUBJ_SYMBOL, subject_label)
    template = template.replace(OBJ_SYMBOL, object_label)
    return [template]


def init_logging(log_directory):
    logger = logging.getLogger("LAMA")
    logger.setLevel(logging.DEBUG)

    os.makedirs(log_directory, exist_ok=True)

    # logging format
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # file handler
    fh = logging.FileHandler(str(log_directory) + "/info.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.propagate = False

    return logger


def batchify(data, batch_size):
    msg = ""
    list_samples_batches = []
    list_sentences_batches = []
    current_samples_batch = []
    current_sentences_batches = []
    c = 0

    # sort to group togheter sentences with similar length
    for sample in sorted(
        data, key=lambda k: len(" ".join(k["masked_sentences"]).split())
    ):
        masked_sentences = sample["masked_sentences"]
        current_samples_batch.append(sample)
        current_sentences_batches.append(masked_sentences)
        c += 1
        if c >= batch_size:
            list_samples_batches.append(current_samples_batch)
            list_sentences_batches.append(current_sentences_batches)
            current_samples_batch = []
            current_sentences_batches = []
            c = 0

    # last batch
    if current_samples_batch and len(current_samples_batch) > 0:
        list_samples_batches.append(current_samples_batch)
        list_sentences_batches.append(current_sentences_batches)

    return list_samples_batches, list_sentences_batches, msg


def batchify_negated(data, batch_size):
    msg = ""
    list_sentences_batches = []
    current_sentences_batches = []
    c = 0

    # sort to group togheter sentences with similar length
    for sample in sorted(
        data, key=lambda k: len(" ".join(k["masked_sentences"]).split())
    ):
        if "negated" in sample:
            masked_sentences = sample["negated"]
            current_sentences_batches.append(masked_sentences)
        else:
            current_sentences_batches.append([""])
        c += 1
        if c >= batch_size:
            list_sentences_batches.append(current_sentences_batches)
            current_sentences_batches = []
            c = 0

    # last batch
    if current_sentences_batches and len(current_sentences_batches) > 0:
        list_sentences_batches.append(current_sentences_batches)

    return list_sentences_batches, msg


def run_thread(arguments):

    msg = ""

    # 1. compute the ranking metrics on the filtered log_probs tensor
    sample_MRR, sample_P, experiment_result, return_msg = metrics.get_ranking(
        arguments["filtered_log_probs"],
        arguments["masked_indices"],
        arguments["vocab"],
        label_index=arguments["label_index"],
        index_list=arguments["index_list"],
        print_generation=arguments["interactive"],
        topk=10000,
    )
    msg += "\n" + return_msg

    sample_perplexity = 0.0
    if arguments["interactive"]:
        pprint(arguments["sample"])
        # THIS IS OPTIONAL - mainly used for debuggind reason
        # 2. compute perplexity and print predictions for the complete log_probs tensor
        sample_perplexity, return_msg = print_sentence_predictions(
            arguments["original_log_probs"],
            arguments["token_ids"],
            arguments["vocab"],
            masked_indices=arguments["masked_indices"],
            print_generation=arguments["interactive"],
        )
        input("press enter to continue...")
        msg += "\n" + return_msg

    return experiment_result, sample_MRR, sample_P, sample_perplexity, msg


def run_thread_negated(arguments):

    msg = ""

    overlap, spearman, return_msg = metrics.get_negation_metric(
        arguments["log_probs"],
        arguments["masked_indices"],
        arguments["log_probs_negated"],
        arguments["masked_indices_negated"],
        arguments["vocab"],
        index_list=arguments["index_list"],
    )

    msg += "\n" + return_msg

    return overlap, spearman, msg


def lowercase_samples(samples, use_negated_probes=False):
    new_samples = []
    for sample in samples:
        sample["obj_label"] = sample["obj_label"].lower()
        sample["sub_label"] = sample["sub_label"].lower()
        lower_masked_sentences = []
        for sentence in sample["masked_sentences"]:
            sentence = sentence.lower()
            sentence = sentence.replace(base.MASK.lower(), base.MASK)
            lower_masked_sentences.append(sentence)
        sample["masked_sentences"] = lower_masked_sentences

        if "negated" in sample and use_negated_probes:
            for sentence in sample["negated"]:
                sentence = sentence.lower()
                sentence = sentence.replace(base.MASK.lower(), base.MASK)
                lower_masked_sentences.append(sentence)
            sample["negated"] = lower_masked_sentences

        new_samples.append(sample)
    return new_samples


def filter_samples(model, samples, vocab_subset, max_sentence_length, template):
    """Filter samples based on vocabulary and length constraints."""
    filtered_samples = []
    excluded_samples = 0
    
    print(f"\nStarting to filter {len(samples)} samples")
    print(f"Model vocab size: {len(model.vocab)}")
    print(f"Vocab subset size: {len(vocab_subset) if vocab_subset else 'None'}")
    
    for sample in samples:
        # Check if answer is in vocabulary
        answer = sample.get('obj_label', '')
        if not answer:
            print(f"Sample {sample.get('uuid', 'unknown')} excluded: Empty answer")
            excluded_samples += 1
            continue
            
        # Check if answer is in vocabulary
        answer_ids = model.get_id(answer)
        if not answer_ids:
            # Try lowercase version
            answer_ids = model.get_id(answer.lower())
            if not answer_ids:
                # Try to split into words
                words = answer.split()
                if all(model.get_id(word) for word in words):
                    answer_ids = [model.get_id(word) for word in words]
                else:
                    print(f"Sample {sample.get('uuid', 'unknown')} excluded: Answer '{answer}' not in vocabulary")
                    excluded_samples += 1
                    continue
            
        # Check masked sentence length and format
        masked_sentences = sample.get('masked_sentences', [])
        if not masked_sentences:
            print(f"Sample {sample.get('uuid', 'unknown')} excluded: No masked sentences")
            excluded_samples += 1
            continue
            
        # Check each masked sentence
        valid_sentences = []
        for sent in masked_sentences:
            if len(sent.split()) <= max_sentence_length:
                valid_sentences.append(sent)
            else:
                print(f"Sample {sample.get('uuid', 'unknown')} excluded: Sentence too long")
                break
                
        if valid_sentences:
            sample['masked_sentences'] = valid_sentences
            filtered_samples.append(sample)
        else:
            excluded_samples += 1
            
    msg = f"Filtering complete. {len(filtered_samples)} samples accepted, {excluded_samples} excluded"
    return filtered_samples, msg


def main(args, shuffle_data=True, model=None):
    print("\nStarting evaluation...")
    print(f"Model type: {args.models_names[0]}")
    print(f"Dataset: {args.dataset_filename}")
    
    # Initialize vocab_subset
    vocab_subset = None
    if args.common_vocab_filename is not None:
        print(f"\nLoading common vocabulary from {args.common_vocab_filename}")
        vocab_subset = load_vocab(args.common_vocab_filename)
        print(f"Loaded {len(vocab_subset)} words from common vocabulary")
    
    # Load and filter samples
    print("\nLoading samples...")
    samples = load_file(args.dataset_filename)
    print(f"Loaded {len(samples)} samples")
    
    # Filter samples based on vocabulary and length
    print("\nFiltering samples...")
    filtered_samples, msg = filter_samples(model, samples, vocab_subset, args.max_sentence_length, args.template)
    print(f"Filtered to {len(filtered_samples)} samples")
    print(msg)
    
    if not filtered_samples:
        print("No valid samples after filtering. Exiting.")
        return

    if len(args.models_names) > 1:
        raise ValueError('Please specify a single language model (e.g., --lm "bert").')

    msg = ""

    [model_type_name] = args.models_names

    print(model)
    if model is None:
        model = build_model_by_name(model_type_name, args)

    if model_type_name == "fairseq":
        model_name = "fairseq_{}".format(args.fairseq_model_name)
    elif model_type_name == "bert":
        model_name = "BERT_{}".format(args.bert_model_name)
    elif model_type_name == "elmo":
        model_name = "ELMo_{}".format(args.elmo_model_name)
    else:
        model_name = model_type_name.title()

    # initialize logging
    if args.full_logdir:
        log_directory = args.full_logdir
    else:
        log_directory = create_logdir_with_timestamp(args.logdir, model_name)
    logger = init_logging(log_directory)
    msg += "model name: {}\n".format(model_name)

    # deal with vocab subset
    index_list = None
    msg += "args: {}\n".format(args)
    if vocab_subset is not None:
        # optimization for some LM (such as ELMo)
        model.optimize_top_layer(vocab_subset)

        filter_logprob_indices, index_list = model.init_indices_for_filter_logprobs(
            vocab_subset, logger
        )

    logger.info("\n" + msg + "\n")

    # dump arguments on file for log
    with open("{}/args.json".format(log_directory), "w") as outfile:
        json.dump(vars(args), outfile)

    # stats
    samples_with_negative_judgement = 0
    samples_with_positive_judgement = 0

    # Mean reciprocal rank
    MRR = 0.0
    MRR_negative = 0.0
    MRR_positive = 0.0

    # Precision at (default 10)
    Precision = 0.0
    Precision1 = 0.0
    Precision_negative = 0.0
    Precision_positivie = 0.0

    # spearman rank correlation
    # overlap at 1
    if args.use_negated_probes:
        Spearman = 0.0
        Overlap = 0.0
        num_valid_negation = 0.0

    # if template is active (1) use a single example for (sub,obj) and (2) ...
    if args.template and args.template != "":
        facts = []
        for sample in filtered_samples:
            sub = sample["sub_label"]
            obj = sample["obj_label"]
            if (sub, obj) not in facts:
                facts.append((sub, obj))
        local_msg = "distinct template facts: {}".format(len(facts))
        logger.info("\n" + local_msg + "\n")
        print(local_msg)
        filtered_samples = []
        for fact in facts:
            (sub, obj) = fact
            sample = {}
            sample["sub_label"] = sub
            sample["obj_label"] = obj
            # substitute all sentences with a standard template
            sample["masked_sentences"] = parse_template(
                args.template.strip(), sample["sub_label"].strip(), base.MASK
            )
            if args.use_negated_probes:
                # substitute all negated sentences with a standard template
                sample["negated"] = parse_template(
                    args.template_negated.strip(),
                    sample["sub_label"].strip(),
                    base.MASK,
                )
            filtered_samples.append(sample)

    # create uuid if not present
    i = 0
    for sample in filtered_samples:
        if "uuid" not in sample:
            sample["uuid"] = i
        i += 1

    # shuffle data
    if shuffle_data:
        shuffle(filtered_samples)

    samples_batches, sentences_batches, ret_msg = batchify(filtered_samples, args.batch_size)
    logger.info("\n" + ret_msg + "\n")
    if args.use_negated_probes:
        sentences_batches_negated, ret_msg = batchify_negated(
            filtered_samples, args.batch_size
        )
        logger.info("\n" + ret_msg + "\n")

    # ThreadPool
    num_threads = args.threads
    if num_threads <= 0:
        # use all available threads
        num_threads = multiprocessing.cpu_count()
    pool = ThreadPool(num_threads)
    list_of_results = []

    for i in tqdm(range(len(samples_batches))):

        samples_b = samples_batches[i]
        sentences_b = sentences_batches[i]

        if args.use_negated_probes:
            sentences_b_negated = sentences_batches_negated[i]
            (
                log_probs_negated,
                token_ids_negated,
                masked_indices_negated,
                generated_answers_negated,
            ) = model.get_batch_generation(sentences_b_negated, logger=logger)

        (
            log_probs,
            token_ids,
            masked_indices,
        ) = model.get_batch_generation(sentences_b, logger=logger)

        # Get gold answers
        gold_answers = [sample["obj_label"] for sample in samples_b]

        # Original evaluation for compatibility
        for i in range(len(sentences_b)):
            sample = samples_b[i]
            sample["token_ids"] = token_ids[i]
            sample["masked_indices"] = masked_indices[i]
            sample["log_probs"] = log_probs[i]

            if args.use_negated_probes:
                sample["token_ids_negated"] = token_ids_negated[i]
                sample["masked_indices_negated"] = masked_indices_negated[i]
                sample["log_probs_negated"] = log_probs_negated[i]

            if vocab_subset is not None:
                # filter log_probs
                filtered_log_probs = model.filter_logprobs(
                    log_probs, token_ids, masked_indices
                )
            else:
                filtered_log_probs = log_probs

            label_index = model.get_id(sample["obj_label"])

            # MAKE SURE THAT obj_label IS IN VOCABULARIES
            if label_index is None:
                raise ValueError(
                    "object label {} not in model vocabulary".format(
                        sample["obj_label"]
                    )
                )
            elif vocab_subset is not None and sample["obj_label"] not in vocab_subset:
                # Try lowercase version in vocab subset
                if sample["obj_label"].lower() not in vocab_subset:
                    raise ValueError(
                        "object label {} not in vocab subset".format(sample["obj_label"])
                    )

            element = {}
            element["sample"] = sample
            element["uuid"] = sample["uuid"]
            element["token_ids"] = token_ids[i]
            element["masked_indices"] = masked_indices[i]
            element["label_index"] = label_index
            element["masked_topk"] = filtered_log_probs[i]
            
            # Calculate MRR and Precision
            if isinstance(filtered_log_probs[i], torch.Tensor):
                probs = torch.exp(filtered_log_probs[i])
                probs = probs / probs.sum()
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                
                # Find rank of correct label
                if isinstance(label_index, (list, tuple)) and len(label_index) > 0:
                    label_idx = label_index[0]
                    rank_matches = (sorted_indices == label_idx).nonzero(as_tuple=True)
                    if len(rank_matches[0]) > 0:
                        rank = rank_matches[0][0].item() + 1
                        element["sample_MRR"] = 1.0 / rank
                        element["sample_Precision"] = 1.0 if rank <= 10 else 0.0
                        element["sample_Precision1"] = 1.0 if rank == 1 else 0.0
                    else:
                        element["sample_MRR"] = 0.0
                        element["sample_Precision"] = 0.0
                        element["sample_Precision1"] = 0.0
                else:
                    element["sample_MRR"] = 0.0
                    element["sample_Precision"] = 0.0
                    element["sample_Precision1"] = 0.0
            else:
                element["sample_MRR"] = 0.0
                element["sample_Precision"] = 0.0
                element["sample_Precision1"] = 0.0

            MRR += element["sample_MRR"]
            Precision += element["sample_Precision"]
            Precision1 += element["sample_Precision1"]

            # the judgment of the annotators recording whether they are
            # evidence in the sentence that indicates a relation between two entities.
            num_yes = 0
            num_no = 0

            if "judgments" in sample:
                # only for Google-RE
                for x in sample["judgments"]:
                    if x["judgment"] == "yes":
                        num_yes += 1
                    else:
                        num_no += 1
                if num_no >= num_yes:
                    samples_with_negative_judgement += 1
                    element["judgement"] = "negative"
                    MRR_negative += element["sample_MRR"]
                    Precision_negative += element["sample_Precision"]
                else:
                    samples_with_positive_judgement += 1
                    element["judgement"] = "positive"
                    MRR_positive += element["sample_MRR"]
                    Precision_positivie += element["sample_Precision"]

            list_of_results.append(element)

    pool.close()
    pool.join()

    # stats
    # Mean reciprocal rank
    MRR /= len(list_of_results)

    # Precision
    Precision /= len(list_of_results)
    Precision1 /= len(list_of_results)

    msg = "all_samples: {}\n".format(len(filtered_samples))
    msg += "list_of_results: {}\n".format(len(list_of_results))
    msg += "global MRR: {}\n".format(MRR)
    msg += "global Precision at 10: {}\n".format(Precision)
    msg += "global Precision at 1: {}\n".format(Precision1)

    if args.use_negated_probes:
        Overlap /= num_valid_negation
        Spearman /= num_valid_negation
        msg += "\n"
        msg += "results negation:\n"
        msg += "all_negated_samples: {}\n".format(int(num_valid_negation))
        msg += "global spearman rank affirmative/negated: {}\n".format(Spearman)
        msg += "global overlap at 1 affirmative/negated: {}\n".format(Overlap)

    if samples_with_negative_judgement > 0 and samples_with_positive_judgement > 0:
        # Google-RE specific
        MRR_negative /= samples_with_negative_judgement
        MRR_positive /= samples_with_positive_judgement
        Precision_negative /= samples_with_negative_judgement
        Precision_positivie /= samples_with_positive_judgement
        msg += "samples_with_negative_judgement: {}\n".format(
            samples_with_negative_judgement
        )
        msg += "samples_with_positive_judgement: {}\n".format(
            samples_with_positive_judgement
        )
        msg += "MRR_negative: {}\n".format(MRR_negative)
        msg += "MRR_positive: {}\n".format(MRR_positive)
        msg += "Precision_negative: {}\n".format(Precision_negative)
        msg += "Precision_positivie: {}\n".format(Precision_positivie)

    logger.info("\n" + msg + "\n")
    print("\n" + msg + "\n")

    # dump pickle with the result of the experiment
    all_results = dict(
        list_of_results=list_of_results, 
        global_MRR=MRR, 
        global_P_at_10=Precision,
        global_P_at_1=Precision1
    )
    with open("{}/result.pkl".format(log_directory), "wb") as f:
        pickle.dump(all_results, f)

    # Print detailed results for each sample
    print("\nDetailed Results for 7 Samples:")
    print("=" * 80)
    
    # Ensure we have enough samples
    if len(list_of_results) < 7:
        print(f"Warning: Only {len(list_of_results)} samples available")
        samples_to_show = len(list_of_results)
    else:
        samples_to_show = 7
    
    for i, result in enumerate(list_of_results[:samples_to_show]):
        print(f"\nSample {i+1}:")
        print("-" * 40)
        
        # Display the actual question with the masked token
        masked_sentence = result['sample']['masked_sentences'][0]
        print(f"Question: {masked_sentence}")
        print(f"Expected Answer: {result['sample']['obj_label']}")
        print(f"MRR: {result['sample_MRR']:.6f}")
        print(f"Precision@1: {result['sample_Precision1']:.6f}")
        print(f"Precision@10: {result['sample_Precision']:.6f}")
        
        # Print top predictions if available
        if 'masked_topk' in result and isinstance(result['masked_topk'], torch.Tensor):
            try:
                # Convert log probabilities to probabilities
                probs = torch.exp(result['masked_topk'])
                probs = probs / probs.sum()
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probs, k=min(10, len(probs)))
                
                print("\nTop 10 Predictions:")
                seen_words = set()  # Track unique words
                for j, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    try:
                        pred_word = model.vocab[idx.item()]
                        # Skip if we've already seen this word
                        if pred_word in seen_words:
                            continue
                        seen_words.add(pred_word)
                        
                        pred_prob = prob.item()
                        is_correct = pred_word.lower() == result['sample']['obj_label'].lower()
                        correct_mark = "✓" if is_correct else "✗"
                        print(f"{j+1}. {pred_word} ({pred_prob:.6f}) {correct_mark}")
                    except Exception as e:
                        print(f"Error processing prediction {j+1}: {str(e)}")
                        continue

                # Print rank of correct answer
                correct_idx = model.get_id(result['sample']['obj_label'])
                if correct_idx and len(correct_idx) > 0:
                    try:
                        # Find the rank of the correct answer
                        correct_rank = (top_indices == correct_idx[0]).nonzero(as_tuple=True)
                        if len(correct_rank[0]) > 0:
                            rank = correct_rank[0][0].item() + 1
                            print(f"\nCorrect answer '{result['sample']['obj_label']}' found at rank {rank}")
                        else:
                            print(f"\nCorrect answer '{result['sample']['obj_label']}' not found in top {len(top_indices)} predictions")
                    except Exception as e:
                        print(f"Error finding correct answer rank: {str(e)}")
                else:
                    print(f"\nCould not find correct answer '{result['sample']['obj_label']}' in vocabulary")
            except Exception as e:
                print(f"Error processing predictions: {str(e)}")
        
        print("\n" + "=" * 80)

    # Calculate final scores
    total_samples = len(list_of_results)
    if total_samples > 0:
        MRR = sum(result['sample_MRR'] for result in list_of_results) / total_samples
        Precision1 = sum(result['sample_Precision1'] for result in list_of_results) / total_samples
        Precision10 = sum(result['sample_Precision'] for result in list_of_results) / total_samples
    else:
        MRR = 0.0
        Precision1 = 0.0
        Precision10 = 0.0

    print("\nSummary Statistics:")
    print(f"Total Samples Processed: {total_samples}")
    print(f"Average MRR: {MRR:.6f}")
    print(f"Average Precision@1: {Precision1:.6f}")
    print(f"Average Precision@10: {Precision10:.6f}")

    return {
        "exact_match": Precision1,
        "f1_score": MRR
    }


if __name__ == "__main__":
    parser = options.get_eval_KB_completion_parser()
    args = options.parse_args(parser)
    main(args)
