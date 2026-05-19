# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import MethodType
from typing import TYPE_CHECKING, Optional

import torch
from transformers import Trainer
from typing_extensions import override

from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomTrainer(Trainer):
    r"""Inherit Trainer for custom optimizer."""

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):

        loss, outputs = super().compute_loss(model, inputs, return_outputs=True,*args, **kwargs)
        # print(outputs.keys())
        balance_loss,lpr_loss,mola_aux_loss = None,None,None
        if "balance_loss" in outputs:
            balance_loss = outputs.balance_loss
        if "lpr_loss" in outputs:
            lpr_loss = outputs.lpr_loss
        if "aux_loss" in outputs:
            mola_aux_loss = outputs.aux_loss

        if mola_aux_loss != None:
            logs: Dict[str, float] = {}
            logs[f"aux_loss"] = mola_aux_loss.item()
            self.log(logs)

        # print(outputs)

        if balance_loss != None:
            nlayers = len(outputs.router_logits)
            prefix = "load_balance_"
            with torch.no_grad():
                mask = inputs['attention_mask'].reshape(-1) # n
                mask = mask.unsqueeze(0).expand(nlayers, mask.size(0)).bool() # nlayers x n
                router_logits = outputs.router_logits
                router_logits = torch.stack(router_logits) # nlayers x n x e
                probs = torch.nn.functional.softmax(router_logits, dim=-1) # nlayers x n x e
                probs = torch.mean(probs[mask], dim=0).detach().cpu() # e

            logs: Dict[str, float] = {}
            logs[f"{prefix}_loss"] = balance_loss.item()
            logs["scores_per_expert"] = " ".join([str(round(k, 2)) for k in probs.tolist()])
            # print(logs)
            self.log(logs)

        if lpr_loss != None:
            prefix = "lpr_"
            lang_mask = inputs['langs']
            with torch.no_grad():
                router_logits = outputs.router_logits
                router_logits = torch.stack(router_logits) # nlayers x n x e
                probs = torch.nn.functional.softmax(router_logits, dim=-1)
                mask = lang_mask.reshape(-1).bool().expand(probs.size()[:2])
                probs = probs[mask].to(torch.float).reshape(mask.size(0), -1, probs.size(-1)) # nlayers x n x e
                probs = probs[:, :, 0] # nlayers x n
                score_expert0 = torch.mean(probs, dim=-1).detach().cpu()

            logs: Dict[str, float] = {}
            logs[f"{prefix}_loss"] = lpr_loss.item()
            logs["old_lang_expert0_score"] = " ".join([str(round(k, 2)) for k in score_expert0.tolist()])
            self.log(logs)

        return loss
