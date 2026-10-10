"""
Optimizer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import copy
import torch
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from .muon_kimi import *

OPTIMIZERS = Registry("optimizers")


OPTIMIZERS.register_module(module=torch.optim.SGD, name="SGD")
OPTIMIZERS.register_module(module=torch.optim.Adam, name="Adam")
OPTIMIZERS.register_module(module=torch.optim.AdamW, name="AdamW")
OPTIMIZERS.register_module(module=MuonKIMI, name="Muon_KIMI")


def build_optimizer(cfg, model, param_dicts=None):
    cfg = copy.deepcopy(cfg)
    if param_dicts is None:
        cfg.params = model.parameters()
    else:
        cfg.params = [dict(names=[], params=[], lr=cfg.lr)]
        module_param_ids = []
        for i in range(len(param_dicts)):
            param_group = dict(names=[], params=[])
            rule = param_dicts[i]
            if "module_type" in rule:
                module_param_ids.append({
                    id(p)
                    for name, module in model.named_modules()
                    if rule.keyword in name
                    and module.__class__.__name__ == rule.module_type
                    for p in module.parameters()
                })
            else:
                module_param_ids.append(None)
            if "lr_scale" in rule:
                if "lr" in rule:
                    raise ValueError("Specify either lr or lr_scale in a parameter group")
                param_group["lr_scale"] = float(rule.lr_scale)
                param_group["lr"] = cfg.lr * param_group["lr_scale"]
            if "lr" in param_dicts[i].keys():
                param_group["lr"] = param_dicts[i].lr
            if "momentum" in param_dicts[i].keys():
                param_group["momentum"] = param_dicts[i].momentum
            if "weight_decay" in param_dicts[i].keys():
                param_group["weight_decay"] = param_dicts[i].weight_decay
            if "force_adamw" in param_dicts[i].keys():
                param_group["force_adamw"] = param_dicts[i].force_adamw
            cfg.params.append(param_group)

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            flag = False
            for i in range(len(param_dicts)):
                matched = (
                    id(p) in module_param_ids[i]
                    if module_param_ids[i] is not None
                    else param_dicts[i].keyword in n
                )
                if matched:
                    cfg.params[i + 1]["names"].append(n)
                    cfg.params[i + 1]["params"].append(p)
                    flag = True
                    break
            if not flag:
                cfg.params[0]["names"].append(n)
                cfg.params[0]["params"].append(p)

        logger = get_root_logger()
        for i in range(len(cfg.params)):
            cfg.params[i].pop("names")
            group_name = "default" if i == 0 else param_dicts[i - 1].keyword
            param_count = sum(param.numel() for param in cfg.params[i]["params"])
            logger.info(
                "Params Group %d (%s): lr=%s weight_decay=%s tensors=%d parameters=%d",
                i + 1,
                group_name,
                cfg.params[i].get("lr", cfg.lr),
                cfg.params[i].get("weight_decay", cfg.weight_decay if "weight_decay" in cfg else None),
                len(cfg.params[i]["params"]),
                param_count,
            )
    return OPTIMIZERS.build(cfg=cfg)
