from __future__ import annotations

"""
Offline rollout loading & merging utilities for verL
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from omegaconf import ListConfig
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin  # type: ignore

import datasets  # huggingface datasets

from verl.protocol import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length


@dataclass
class _RawOfflineResponse:
    text: str
    log_probs: Sequence[float]


class OfflineRolloutDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_files: str | Sequence[str] | ListConfig,
        online_tokenizer: PreTrainedTokenizer,
        offline_tokenizer: PreTrainedTokenizer,
        cfg,
        processor: ProcessorMixin | None = None,
    ) -> None:
        if not isinstance(data_files, (Sequence, ListConfig)) or isinstance(
            data_files, str
        ):
            data_files = [data_files]
        self.online_tokenizer = online_tokenizer
        self.offline_tokenizer = offline_tokenizer
        self.config = cfg
        self.processor = processor

        ds = datasets.load_dataset("parquet", data_files=data_files)["train"]
        df = ds.select_columns(["extra_info", "responses", "rollout_log_probs"]).to_pandas()
        df["index"] = df["extra_info"].apply(lambda x: int(x["index"]))
        df_exploded = df.explode(["responses", "rollout_log_probs"])
        df_exploded["offline_response"] = [
            _RawOfflineResponse(text=txt, log_probs=lp)
            for txt, lp in zip(df_exploded["responses"], df_exploded["rollout_log_probs"])
        ]
        self._idx2rows = df_exploded.groupby("index")["offline_response"].apply(list).to_dict()
        self._indices: List[int] = sorted(self._idx2rows.keys())

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int):
        idx = self._indices[i]
        return idx, self._idx2rows[idx]

    def get_by_index(self, index: int, n_offline: int) -> List[_RawOfflineResponse]:
        """Get raw responses by index."""
        items = self._idx2rows.get(index, [])
        # random select n_offline responses if there are more than n_offline responses
        if len(items) > n_offline:
            items = random.sample(items, n_offline)
        return items, len(items)

    def get_by_indices(self, indices: List[int], n_offline: int) -> List[_RawOfflineResponse]:
        # combine responses from multiple indices to a single list
        """Get raw responses by indices."""
        responses = []
        num_of_responses = []
        for idx in indices:
            items, num_items = self.get_by_index(idx, n_offline)
            responses.extend(items)
            num_of_responses.append(num_items)
        return responses, num_of_responses

    def build_dataproto(
            self,
            batch_dict: Dict,
            n_offline: int,
    ) -> DataProto | None:
        indices = batch_dict["index"]
        idx = batch_dict["input_ids"]
        position_ids = batch_dict["position_ids"]
        attention_mask = batch_dict["attention_mask"]

        offline_response_items, offline_num_of_responses= self.get_by_indices(indices, n_offline)
        if not offline_response_items or len(offline_response_items) == 0:
            return None

        offline_batch_size = len(offline_response_items)
        assert sum(offline_num_of_responses) == offline_batch_size, "The number of responses should match the batch size."

        offline_response_texts = [resp.text for resp in offline_response_items]
        offline_response_tokens = [self.offline_tokenizer(offline_response_text, add_special_tokens=False)["input_ids"] for offline_response_text in offline_response_texts]
        # TODO : check the correctness the offset of the log probs 
        offline_response_tokens = [input_ids[2:] for input_ids in offline_response_tokens]
        offline_responses_log_probs = [resp.log_probs[2:-1] for resp in offline_response_items]

        response = pad_2d_list_to_length(offline_response_tokens, self.online_tokenizer.pad_token_id, max_length=self.config.data.max_response_length).to(idx.device)
        rollout_log_probs = pad_2d_list_to_length(offline_responses_log_probs, -1, max_length=self.config.data.max_response_length).to(idx.device)
        rollout_log_probs = rollout_log_probs.to(torch.float32)

        repeats_tensor = torch.tensor(offline_num_of_responses, dtype=torch.long)
        idx = torch.repeat_interleave(idx, repeats=repeats_tensor, dim=0)
        position_ids = torch.repeat_interleave(position_ids, repeats=repeats_tensor, dim=0)
        attention_mask = torch.repeat_interleave(attention_mask, repeats=repeats_tensor, dim=0)

        seq = torch.cat([idx, response], dim=-1)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(offline_batch_size, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(response_id=response, eos_token=self.online_tokenizer.eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "rollout_log_probs": rollout_log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=offline_batch_size,
        )

        non_tensor_batch = {
            "data_source": batch_dict["data_source"],
            "ability": batch_dict["ability"],
            "reward_model": batch_dict["reward_model"],
            "extra_info": batch_dict["extra_info"],
            "index": batch_dict["index"],
            "uid": batch_dict["uid"],
            "tools_kwargs": batch_dict["tools_kwargs"],
        }
        repeated_non_tensor_batch = {}

        for key, val in non_tensor_batch.items():
            repeated_non_tensor_batch[key] = np.repeat(val, offline_num_of_responses, axis=0)

        return DataProto(batch=batch, non_tensor_batch=repeated_non_tensor_batch)
