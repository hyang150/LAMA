# File: lama/lama/evaluation_metrics.py

import torch
import numpy as np
import scipy
import scipy.stats

def __max_probs_values_indices(masked_indices, log_probs, topk=1000):
    """
    Safely extract the log-probs row for the first masked index, handle 1-D inputs,
    and guard against out-of-bounds masked indices.
    Returns:
      selected (Tensor [vocab_size]), index_max_probs (ndarray [topk]),
      value_max_probs (ndarray [topk])
    """
    # only consider the first mask
    masked_indices = masked_indices[:1]

    # normalize to list
    if isinstance(masked_indices, int):
        masked_indices = [masked_indices]
    masked_index = masked_indices[0]

    # if log_probs is 1-D (vocab only), unsqueeze to [1, vocab]
    if log_probs.dim() == 1:
        log_probs = log_probs.unsqueeze(0)

    seq_len = log_probs.size(0)

    # if mask position is out-of-bounds, return empty results
    if masked_index >= seq_len or masked_index < 0:
        return (
            torch.empty(0, dtype=log_probs.dtype, device=log_probs.device),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    # select the row for the masked position
    selected = log_probs[masked_index]  # shape: [vocab_size]

    # get topk
    value_max_probs, index_max_probs = torch.topk(selected, k=topk, dim=0)
    index_max_probs = index_max_probs.cpu().numpy().astype(int)
    value_max_probs = value_max_probs.detach().cpu().numpy()

    return selected, index_max_probs, value_max_probs


def __print_top_k(value_max_probs, index_max_probs, vocab, mask_topk, index_list, max_printouts=10):
    """
    Build a human-readable top-k list from numpy arrays.
    Safely handles empty or too-short index_list.
    """
    result = []

    # no predictions
    if index_max_probs.size == 0:
        return result, "No valid predictions for this masked token (out-of-bounds).\n"

    # only iterate over the available number of predictions
    n = min(mask_topk, len(index_max_probs))
    msg = "\n| Top{} predictions\n".format(min(max_printouts, n))
    msg += "-" * 50 + "\n"
    msg += "{:<8s}{:<20s}{:<12s}\n".format("rank", "token", "log_prob")
    msg += "-" * 50 + "\n"

    for i in range(n):
        filtered_idx = int(index_max_probs[i])

        # map through index_list if valid, otherwise raw
        if isinstance(index_list, list) and filtered_idx < len(index_list):
            idx = index_list[filtered_idx]
        else:
            idx = filtered_idx

        log_prob = float(value_max_probs[i])
        word_form = vocab[idx]

        if i < max_printouts:
            msg += "{:<8d}{:<20s}{:<12.3f}\n".format(i + 1, word_form, log_prob)

        result.append({
            'rank': i + 1,
            'token_idx': idx,
            'log_prob': log_prob,
            'token_word_form': word_form
        })

    return result, msg


def get_ranking(log_probs, masked_indices, vocab, label_index=None,
                index_list=None, topk=1000, P_AT=10, print_generation=True):
    """
    Compute ranking metrics for a single example.
    Returns: MRR, P@X, dict of results, optionally prints top-k.
    """
    experiment_result = {}

    # extract top-k candidates
    sel, index_max_probs, value_max_probs = __max_probs_values_indices(
        masked_indices, log_probs, topk=topk)
    result_masked_topk, return_msg = __print_top_k(
        value_max_probs, index_max_probs, vocab, topk, index_list)
    experiment_result['topk'] = result_masked_topk

    if print_generation:
        print(return_msg)

    MRR = 0.0
    P_AT_X = 0.0
    P_AT_1 = 0.0
    PERPLEXITY = None

    if label_index is not None:
        # map label_index through index_list if needed
        if isinstance(index_list, list) and label_index in index_list:
            label_index_mapped = index_list.index(label_index)
        else:
            label_index_mapped = label_index

        # find rank of the true label
        positions = np.where(index_max_probs == label_index_mapped)[0]
        if positions.size > 0:
            rank = int(positions[0]) + 1
            MRR = 1.0 / rank
            if rank <= P_AT:
                P_AT_X = 1.0
            if rank == 1:
                P_AT_1 = 1.0

        # compute perplexity for the true label
        if sel.numel() > 0:
            try:
                PERPLEXITY = float(sel[label_index_mapped])
            except IndexError:
                PERPLEXITY = None

    experiment_result["MRR"] = MRR
    experiment_result["P_AT_X"] = P_AT_X
    experiment_result["P_AT_1"] = P_AT_1
    experiment_result["PERPLEXITY"] = PERPLEXITY

    return MRR, P_AT_X, experiment_result, return_msg


def __overlap_negation(index_max_probs_negated, index_max_probs):
    """
    Compare top-1 predictions of affirmative vs negated sentences.
    """
    return int(index_max_probs_negated[0] == index_max_probs[0])


def get_negation_metric(log_probs, masked_indices,
                        log_probs_negated, masked_indices_negated,
                        vocab, index_list=None, topk=1):
    """
    Compute overlap and Spearman correlation between affirmative and negated log-probs.
    """
    if len(masked_indices_negated) == 0:
        return np.nan, np.nan, ""

    # get top-1 for both
    _, idx_aff, _ = __max_probs_values_indices(masked_indices, log_probs, topk=topk)
    _, idx_neg, _ = __max_probs_values_indices(masked_indices_negated,
                                              log_probs_negated, topk=topk)

    overlap = __overlap_negation(idx_neg, idx_aff)
    # spearman on the entire distributions
    sp_corr = scipy.stats.spearmanr(
        log_probs.flatten().cpu().numpy(),
        log_probs_negated.flatten().cpu().numpy()
    ).correlation

    return overlap, sp_corr, ""
