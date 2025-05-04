# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lama.modules.base_connector import Base_Connector, OPENAI_EOS
from collections import Counter

# Pattern for ASCII tokens (used if filtering needed)
_ascii_pattern = re.compile(r'^[\x00-\x7F]+$')

class GPTJ(Base_Connector):
    def __init__(self, args):
        super().__init__()
        model_name = getattr(args, 'gptj_model_name', 'EleutherAI/gpt-j-6B')
        self.mask_token = getattr(args, 'mask_token', '[MASK]')

        # Load tokenizer and ensure mask token exists
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.mask_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({'mask_token': self.mask_token})

        # Load model and resize embeddings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

        # Build LAMA-style vocab
        id_to_tok = {v: k for k, v in self.tokenizer.get_vocab().items()}
        def convert_word(tok):
            if tok.startswith('Ġ'):
                return tok[1:]
            if len(tok) == 1:
                return tok
            return f'##{tok}'
        self.vocab = [convert_word(id_to_tok[i]) for i in range(len(id_to_tok))]
        self._init_inverse_vocab()

        # Append common vocab if provided
        common_file = getattr(args, 'common_vocab_filename', None)
        if common_file:
            with open(common_file, 'r', encoding='utf-8') as f:
                for line in f:
                    tok = line.strip()
                    if tok and tok not in self.inverse_vocab:
                        self.inverse_vocab[tok] = len(self.vocab)
                        self.vocab.append(tok)

        # Special token IDs
        self.mask_token_id = self.tokenizer.mask_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.unk_symbol = OPENAI_EOS

        # Initialize word piece tokenizer for out-of-vocabulary words
        self.word_piece_tokenizer = self.tokenizer

    def _cuda(self):
        self.model.cuda()

    def get_id(self, string):
        """Get token IDs for a string, handling out-of-vocabulary words."""
        if not string:
            return None
            
        # First try direct tokenization
        try:
            tokens = self.tokenizer.tokenize(string)
            if tokens:
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                return ids
                
            # Try lowercase version
            tokens = self.tokenizer.tokenize(string.lower())
            if tokens:
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                return ids
                
            # Try splitting into words
            words = string.split()
            if len(words) > 1:
                ids = []
                for word in words:
                    word_tokens = self.tokenizer.tokenize(word)
                    if word_tokens:
                        word_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
                        ids.extend(word_ids)
                    else:
                        # Try lowercase for each word
                        word_tokens = self.tokenizer.tokenize(word.lower())
                        if word_tokens:
                            word_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
                            ids.extend(word_ids)
                if ids:
                    return ids
                    
            # If all else fails, try to find the closest match
            for word in self.vocab:
                if word.lower() == string.lower():
                    return [self.inverse_vocab[word]]
                    
            return None
        except Exception as e:
            print(f"Warning: Error tokenizing '{string}': {str(e)}")
            return None

    def __get_input_tensors(self, sentence):
        # Tokenize single prompt string and record mask positions
        tokenized = []
        masked_indices = []
        parts = sentence.split(self.mask_token)
        for idx, part in enumerate(parts):
            if idx > 0:
                masked_indices.append(len(tokenized))
                tokenized.append(self.mask_token)
            part = part.strip()
            if part:
                tokenized.extend(self.tokenizer.tokenize(part))
        token_ids = self.tokenizer.convert_tokens_to_ids(tokenized)
        indexed = [self.eos_id] + token_ids
        src = torch.tensor(indexed[:-1], dtype=torch.long)
        dst = torch.tensor(indexed[1:], dtype=torch.long)
        return src, dst, masked_indices

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True):
        if try_cuda:
            self.try_cuda()
            
        # Flatten lists to strings
        seqs = [" ".join(s) if isinstance(s, (list, tuple)) else s for s in sentences_list]
        inputs = [self.__get_input_tensors(sent) for sent in seqs]
        src_list, dst_list, mask_list = zip(*inputs)
        src_batch = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True)
        
        with torch.no_grad():
            # Get logits for each position
            outputs = self.model(
                input_ids=src_batch.to(self._model_device),
                return_dict=True
            )
            
            # Get log probabilities for each token
            logits = outputs.logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Process each sequence
            log_probs_list = []
            token_ids_list = []
            masked_indices_list = []
            
            for i, (src, dst, masks) in enumerate(zip(src_list, dst_list, mask_list)):
                # Get log probabilities for this sequence
                seq_log_probs = log_probs[i, :len(src)]
                
                # Get predictions for each mask position
                mask_predictions = []
                for mask_pos in masks:
                    if mask_pos < len(seq_log_probs):
                        mask_probs = seq_log_probs[mask_pos]
                        probs = torch.exp(mask_probs)
                        probs = probs / probs.sum()
                        top_probs, top_indices = torch.topk(probs, k=10)
                        mask_predictions.append((top_probs, top_indices))
                
                # Store results
                log_probs_list.append(seq_log_probs)
                token_ids_list.append(dst)
                masked_indices_list.append(masks)
                
                # Print detailed predictions for debugging
                if logger:
                    logger.info(f"\nProcessing sentence: {seqs[i]}")
                    for j, (probs, indices) in enumerate(mask_predictions):
                        logger.info(f"\nTop 10 predictions at mask position {j}:")
                        for k in range(len(probs)):
                            pred_idx = indices[k].item()
                            pred_word = self.vocab[pred_idx]
                            pred_prob = probs[k].item()
                            logger.info(f"{k+1}. {pred_word}: {pred_prob:.6f}")
            
            return log_probs_list, token_ids_list, masked_indices_list

    def evaluate_qa(self, generated_answers, gold_answers):
        """Evaluate QA performance using exact match and F1 score"""
        from nltk.tokenize import word_tokenize
        import numpy as np
        
        def normalize_answer(s):
            """Lower text and remove punctuation, articles and extra whitespace."""
            import string
            import re
            def remove_articles(text):
                return re.sub(r'\b(a|an|the)\b', ' ', text)
            def white_space_fix(text):
                return ' '.join(text.split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            return white_space_fix(remove_articles(remove_punc(lower(s))))

        def f1_score(prediction, ground_truth):
            prediction_tokens = normalize_answer(prediction).split()
            ground_truth_tokens = normalize_answer(ground_truth).split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                return 0
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        def exact_match_score(prediction, ground_truth):
            return normalize_answer(prediction) == normalize_answer(ground_truth)

        exact_match = []
        f1 = []
        
        for pred, gold in zip(generated_answers, gold_answers):
            exact_match.append(exact_match_score(pred, gold))
            f1.append(f1_score(pred, gold))
            
        return {
            "exact_match": np.mean(exact_match),
            "f1_score": np.mean(f1)
        }

    def filter_logprobs(self, log_probs, token_ids_list=None, masked_list=None):
        """Filter log probabilities to only include valid tokens."""
        if token_ids_list is None or masked_list is None:
            return log_probs
            
        filtered_log_probs = []
        for i, (log_prob, token_ids, masked_indices) in enumerate(zip(log_probs, token_ids_list, masked_list)):
            # Get valid token IDs
            valid_ids = set()
            for token_id in token_ids:
                if token_id < len(self.vocab):
                    valid_ids.add(token_id)
            
            # Filter log probabilities
            mask_log_probs = []
            for mask_pos in masked_indices:
                if mask_pos < len(log_prob):
                    mask_prob = log_prob[mask_pos]
                    # Only keep probabilities for valid tokens
                    valid_probs = torch.full_like(mask_prob, float('-inf'))
                    for valid_id in valid_ids:
                        valid_probs[valid_id] = mask_prob[valid_id]
                    mask_log_probs.append(valid_probs)
            
            if mask_log_probs:
                filtered_log_probs.append(torch.stack(mask_log_probs))
            else:
                filtered_log_probs.append(torch.tensor([]))
        
        return filtered_log_probs