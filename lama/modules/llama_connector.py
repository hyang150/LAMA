# llama_connector.py
# Connector for HuggingFace LLaMA models (e.g., Llama-3.1-8b)
import os
import torch
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer
from lama.modules.base_connector import Base_Connector, OPENAI_EOS

class Llama(Base_Connector):
    def __init__(self, args):
        super().__init__()
        # 优先使用本地目录，否则回退到远端 HF Hub
        local_dir = args.llama_model_dir
        hf_name = args.llama_model_name
        if local_dir and os.path.isdir(local_dir):
            model_source = local_dir
            from_pretrained_kwargs = {"local_files_only": True}
            print(f"Loading LLaMA from local path: {model_source}")
        else:
            model_source = hf_name
            from_pretrained_kwargs = {"local_files_only": False}
            print(f"Downloading LLaMA from HF Hub: {model_source}")

        # Load tokenizer and model
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_source,
            **from_pretrained_kwargs
        )
        self.model = LlamaForCausalLM.from_pretrained(
            model_source,
            **from_pretrained_kwargs
        ).eval()

        # Set device and move model
        self._model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self._model_device)

        # Build vocab and inverse vocab
        self.vocab = [
            self.tokenizer.convert_ids_to_tokens(i)
            for i in range(self.tokenizer.vocab_size)
        ]
        self._init_inverse_vocab()

        # Special tokens
        self.eos_id = self.tokenizer.eos_token_id
        self.unk_symbol = self.tokenizer.unk_token
        self.model_vocab = self.vocab

    def _cuda(self):
        self.model.cuda()

    def get_id(self, string):
        """Convert text to token IDs (no special tokens)."""
        return self.tokenizer.encode(string, add_special_tokens=False)

    def __get_input_tensors(self, sentence_list):
        """
        Turn list of sentences (with [MASK]) into src/dst tensors and mask positions.
        Returns: src_tensor, dst_tensor, masked_indices, tokenized_list
        """
        tokenized = []
        masked_indices = []
        for idx, sent in enumerate(sentence_list):
            if idx > 0:
                tokenized.append(self.eos_id)
            for j, chunk in enumerate(sent.split('[MASK]')):
                if j > 0:
                    masked_indices.append(len(tokenized))
                    tokenized.append(self.unk_symbol)
                if chunk.strip():
                    tokenized.extend(
                        self.tokenizer.encode(chunk.strip(), add_special_tokens=False)
                    )
        full = [self.eos_id] + tokenized
        src = torch.tensor(full[:-1], dtype=torch.long)
        dst = torch.tensor(full[1:], dtype=torch.long)
        return src, dst, masked_indices, tokenized

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True):
        if try_cuda:
            self.try_cuda()
        batch = [self.__get_input_tensors(s) for s in sentences_list]
        src_list, dst_list, masked_list, _ = zip(*batch)
        src_batch = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True)

        with torch.no_grad():
            outputs = self.model(src_batch.to(self._model_device)).logits
            logits = outputs[..., : self.tokenizer.vocab_size]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu()

        token_ids = [dst.numpy() for dst in dst_list]
        return log_probs, token_ids, masked_list

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):
        if try_cuda:
            self.try_cuda()
        batch = [self.__get_input_tensors(s) for s in sentences_list]
        src_list, _, masked_list, _ = zip(*batch)
        src_batch = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True)

        with torch.no_grad():
            out = self.model.model(
                input_ids=src_batch.to(self._model_device),
                output_hidden_states=True
            )
        hidden = out.hidden_states[-1].cpu()
        return [hidden], None, None
