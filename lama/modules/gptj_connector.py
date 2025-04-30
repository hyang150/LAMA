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

    def _cuda(self):
        self.model.cuda()

    def get_id(self, string):
        tokens = self.tokenizer.tokenize(string)
        return self.tokenizer.convert_tokens_to_ids(tokens)

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
            outputs = self.model(src_batch.to(self._model_device))
            logits = outputs.logits[..., :self.model.config.vocab_size]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu()

        log_probs_list = [log_probs[i] for i in range(log_probs.size(0))]
        token_ids_list = [dst_list[i].cpu().numpy() for i in range(len(dst_list))]
        return log_probs_list, token_ids_list, list(mask_list)

    def filter_logprobs(self, log_probs, token_ids_list=None, masked_list=None):
        # Two-arg call: skip filtering and return full log_probs list
        if masked_list is None:
            return log_probs
        # Three-arg call: extract per-mask vectors and ground-truth indices
        original, gt_indices = [], []
        for lp, toks, masks in zip(log_probs, token_ids_list, masked_list):
            for p in masks:
                if 0 <= p < lp.size(0):
                    original.append(lp[p].numpy())
                    gt_indices.append(int(toks[p]))
        return original, gt_indices