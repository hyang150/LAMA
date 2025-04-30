# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from .bert_connector import Bert
# from .elmo_connector import Elmo
from .gpt_connector import GPT
from .gptj_connector import GPTJ
# from .transformerxl_connector import TransformerXL
from .roberta_connector import Roberta
from .llama_connector import Llama


def build_model_by_name(lm, args, verbose=True):
    """Load a model by name and args.

    Note, args.lm is not used for model selection. args are only passed to the
    model's initializer.
    """
    MODEL_NAME_TO_CLASS = dict(
        # elmo=Elmo,
        bert=Bert,
        gpt=GPT,
        gptj=GPTJ,
        # transformerxl=TransformerXL,
        roberta=Roberta,
        llama=Llama
    )
    if lm not in MODEL_NAME_TO_CLASS:
        raise ValueError(f"Unrecognized Language Model: {lm}.")
    if verbose:
        print(f"Loading {lm} model...")
    return MODEL_NAME_TO_CLASS[lm](args)
