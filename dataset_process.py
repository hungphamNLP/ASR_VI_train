from datasets import load_dataset
import pandas as pd
import re
import soundfile as sf
import torch
import numpy as np
import sys
from torch.utils.data import DataLoader
from build_config import processor,config
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2Processor
from transformers import default_data_collator


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = sf.read(batch["file"])
    batch["speech"] = speech_array
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["text"]
    return batch

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    batch["text"] = re.sub('<unk>','',batch["text"])
    return batch

def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    # assert (
    #     len(set(batch["sampling_rate"])) == 1
    # ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=16000).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    

timit_train = load_dataset('json',data_files=config['dataset_train'],cache_dir='./my_cache')
timit_test =  load_dataset('json',data_files=config['dataset_test'],cache_dir='./my_cache')

timit_train = timit_train.map(remove_special_characters)
timit_test = timit_test.map(remove_special_characters)

timit_train = timit_train.map(speech_file_to_array_fn, remove_columns=timit_train.column_names["train"], num_proc=config['num_proc'])
timit_test = timit_test.map(speech_file_to_array_fn, remove_columns=timit_test.column_names["train"], num_proc=config['num_proc'])


timit_train = timit_train.map(prepare_dataset, remove_columns=timit_train.column_names["train"], num_proc=8)
timit_test = timit_test.map(prepare_dataset,remove_columns=timit_test.column_names["train"],num_proc=8)
max_input_length_in_sec = config['max_input_length_in_sec']

timit_train["train"] = timit_train["train"].filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])
timit_test["train"] = timit_test["train"].filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])

# timit_train.set_format("torch")
# timit_test.set_format("torch")

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# train_dataloader = DataLoader(timit_train['train'], shuffle=True,collate_fn=data_collator, batch_size=config['batch'])
# eval_dataloader = DataLoader(timit_test["train"],collate_fn=data_collator,batch_size=config['batch'])