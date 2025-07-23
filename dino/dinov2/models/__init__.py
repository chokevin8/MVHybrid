# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits
from vim import models_mamba as vim
import sys
sys.path.append("/scratch/x2946a13/miccai/")
from MambaVision.mambavision.models import mamba_vision_flat as mamba_vision
from MambaVision.mambavision.models import mamba_vision_flat_fr as frmlp_mamba_vision

from hydra.hydra.modules import hydra as hydramodule
# from hydra.hydra.modules import hydra_conv as hydraconvmodule
import torch.nn as nn
from functools import partial

logger = logging.getLogger("dinov2")

# args = cfg.student
def build_model(args, only_teacher=False, img_size=224): 
    args.arch = args.arch.removesuffix("_memeff")
    print(f"args arch is {args.arch}")
    if not (args.arch.startswith("vit") or args.arch.startswith("vim") or args.arch.startswith("mamba_vision") or args.arch.startswith("hydra") or args.arch.startswith("frmlp")):
        raise ValueError(f"Unsupported architecture: {args.arch}. Expected architectures starting with 'vit' or 'vim' or 'mamba_vision' or 'hydra'.")

    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim # 

    # added vision mamba
    if "vim" in args.arch:

        vim_kwargs = dict(
            img_size=img_size,
            if_einfft=args.if_einfft,
            if_mlp=args.if_mlp,
            if_registers=args.if_registers,
            num_cls_tokens=args.num_cls_tokens,
            cls_reduce=args.cls_reduce
         )
        teacher = vim.__dict__[args.arch](**vim_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vim.__dict__[args.arch](
            **vim_kwargs,
            drop_path_rate=args.drop_path_rate,
        )
        embed_dim = student.embed_dim

    # added mambavision
    if "mamba_vision" in args.arch:

        mamba_vision_kwargs = dict(
            if_einfft=args.if_einfft,
            einfft_mamba_only=args.einfft_mamba_only
        )
        teacher = mamba_vision.__dict__[args.arch](**mamba_vision_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = mamba_vision.__dict__[args.arch](
            **mamba_vision_kwargs,
            drop_path_rate=args.drop_path_rate,
        )
        embed_dim = student.embed_dim

    # added hydravision
    if "hydra" in args.arch:
        hydra_kwargs = dict(img_size=img_size)
        teacher = hydramodule.__dict__[args.arch](**hydra_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = hydramodule.__dict__[args.arch](
            **hydra_kwargs,
            drop_path_rate=args.drop_path_rate,
        )
        embed_dim = student.embed_dim

    if "frmlp" in args.arch:
        mamba_vision_kwargs = dict(img_size=img_size)
        teacher = frmlp_mamba_vision.__dict__[args.arch](**mamba_vision_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = frmlp_mamba_vision.__dict__[args.arch](
            **mamba_vision_kwargs,
            drop_path_rate=args.drop_path_rate,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim

def build_model_from_cfg(cfg, only_teacher=False):
        return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)
