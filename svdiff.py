'''
modified from https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py
Copyright [2022] [kohya-ss]

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
'''

# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
import os
from typing import List, Tuple, Union
import numpy as np
import torch
import re
import torch.nn.utils.parametrize as parametrize


RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")

class SVDiffParametrization(torch.nn.Module):
    def forward(self, d):
        o = torch.mul(self.U, torch.nn.functional.relu(self.S + d)) @ self.Vh
        o = o.reshape(*(d for i, d in enumerate(self.orig_shape) if i != 1), -1)
        m = len(self.orig_shape)
        return o.permute(0, m - 1, *range(1, m-1))
    

    def right_inverse(self, X: torch.Tensor):
        self.orig_shape = X.shape
        X = X.permute(*(i for i, _ in enumerate(X.shape) if i != 1), 1)
        X = X.reshape(-1, self.orig_shape[1])
        self.U, self.S, self.Vh = torch.linalg.svd(X, full_matrices=False)
        return torch.zeros_like(self.S)

class SVDiffModule(torch.nn.Module):
    def __init__(self, lora_name, org_module: torch.nn.Module):
        super().__init__()
        self.lora_name = lora_name
        self.org_module = org_module

    def apply_to(self):
        self.org_module.weight = self.org_module.weight.float()
        parametrize.register_parametrization(self.org_module, "weight", SVDiffParametrization())
    
    def forward(self, x):
        return self.org_module(x)

def create_network(_multiplier, _network_dim, _network_alpha, vae, text_encoder, unet, **kwargs):
    return LoRANetwork(text_encoder, unet)

class LoRANetwork(torch.nn.Module):
    NUM_OF_BLOCKS = 12  # フルモデル相当でのup,downの層の数

    # is it possible to apply conv_in and conv_out? -> yes, newer LoCon supports it (^^;)
    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def __init__(
        self,
        text_encoder,
        unet,
    ) -> None:
        super().__init__()

        # create module instances
        def create_modules(is_unet, root_module: torch.nn.Module, target_replace_modules) -> List[SVDiffModule]:
            prefix = LoRANetwork.LORA_PREFIX_UNET if is_unet else LoRANetwork.LORA_PREFIX_TEXT_ENCODER
            loras = [] # not loras but svdiff modules
            skipped = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            lora = SVDiffModule(lora_name, child_module)
                            loras.append(lora)
            return loras, skipped

        self.text_encoder_loras, skipped_te = create_modules(False, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
        print(f"create SVDiff for Text Encoder: {len(self.text_encoder_loras)} modules.")

        # extend U-Net target modules if conv2d 3x3 is enabled, or load from weights
        target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE
        target_modules += LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        self.unet_loras, skipped_un = create_modules(True, unet, target_modules)
        print(f"create SVDiff for U-Net: {len(self.unet_loras)} modules.")

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(True)
        all_params = []

        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(p for n, p in lora.named_parameters() if 'bias' not in n)
            return params

        if self.text_encoder_loras:
            param_data = {"params": enumerate_params(self.text_encoder_loras)}
            if text_encoder_lr is not None:
                param_data["lr"] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            param_data = {"params": enumerate_params(self.unet_loras)}
            if unet_lr is not None:
                param_data["lr"] = unet_lr
            all_params.append(param_data)

        return all_params

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        # convert from fp16 to fp32 because amp scaler doesn't like fp16s
        for lora in [*self.text_encoder_loras, *self.unet_loras]:
            lora.to(torch.float)
        if 
        text_encoder.requires_grad_(True)
        unet.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)


# 外部から呼び出す可能性を考慮しておく
def get_block_index(lora_name: str) -> int:
    block_idx = -1  # invalid lora name

    m = RE_UPDOWN.search(lora_name)
    if m:
        g = m.groups()
        i = int(g[1])
        j = int(g[3])
        if g[2] == "resnets":
            idx = 3 * i + j
        elif g[2] == "attentions":
            idx = 3 * i + j
        elif g[2] == "upsamplers" or g[2] == "downsamplers":
            idx = 3 * i + 2

        if g[0] == "down":
            block_idx = 1 + idx  # 0に該当するLoRAは存在しない
        elif g[0] == "up":
            block_idx = LoRANetwork.NUM_OF_BLOCKS + 1 + idx

    elif "mid_block_" in lora_name:
        block_idx = LoRANetwork.NUM_OF_BLOCKS  # idx=12

    return block_idx
