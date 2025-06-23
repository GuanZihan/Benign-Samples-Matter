# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List
from ft_datasets.utils import ConcatDataset
import pandas as pd

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
# SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
# SYSTEM_PROMPT = B_SYS + SYSTEM_PROMPT + E_SYS

PROMPT_TEMPLATE = (B_SYS + "You are an intelligent clinical languge model. Below is a snippet of patient's discharge summary and a following instruction from healthcare professional. Write a response that appropriately completes the instruction. The response should provide the accurate answer to the instruction, while being concise. " +
         E_SYS +
        "### Discharge Summary:\n{note}\n\nInstruction:\n{question}\n\n### Response:\n")

def get_mimic_dataset(dataset_config, tokenizer, partition="train", max_words=30, concat=False):
    if concat:
        return ConcatDataset(InstructionDataset(dataset_config, tokenizer, partition, max_words, pad=False))
    else:
        return InstructionDataset(dataset_config, tokenizer, partition, max_words, pad=True)


class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30, pad=True):
        file_path = dataset_config.data_path
        # "starmpcc/Asclepius-Synthetic-Clinical-Notes"
        self.ann = pd.read_csv(file_path).to_dict('records')
        print("load successful!")

        if partition == "train":
            self.ann = self.ann[0:2000]
        else:
            self.ann = self.ann[-200:]

        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad

    def _determine_file_format(self, file_path):
        if file_path.endswith('.jsonl'):
            return 'jsonl'
        elif file_path.endswith('.json'):
            return 'json'
        else:
            raise ValueError("File extension must be .json or .jsonl")

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.ann[index]
        prompt = B_INST + " " + PROMPT_TEMPLATE.format_map(ann) + " " + E_INST
        example = prompt + " " + ann["answer"] + " "
        
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        
        if self.pad:
            padding = self.max_words - example.shape[0]
            if padding > 0:
                example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                example = example[: self.max_words]
        
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
