"""Check backbone Block selection and the shared OneCycle learning-rate curve."""
import copy
import unittest
from pathlib import Path

import torch
from torch import nn

from pointcept.models.point_transformer_v3.point_transformer_v3m3_utonia import (
    Block,
    PointTransformerV3,
)
from pointcept.utils.config import Config, ConfigDict
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler


CONFIG = Path(__file__).resolve().parents[1] / "configs/human_pose/hmr-utonia-v1m1-0-bedlam.py"


class LearningRateTests(unittest.TestCase):
    def make_model(self, lora=False):
        model = nn.Module()
        model.backbone = PointTransformerV3(
            in_channels=9, stride=(2,), enc_depths=(1, 1),
            enc_channels=(12, 24), enc_num_head=(2, 2),
            enc_patch_size=(4, 4), enc_mode=True, enable_flash=False,
        )
        model.head = nn.Linear(24, 3)
        if lora:
            from peft import LoraConfig, get_peft_model

            model.backbone.enc = get_peft_model(
                model.backbone.enc, LoraConfig(r=2, target_modules=["qkv"])
            )
            model.backbone.embedding.requires_grad_(False)
            for name, param in model.backbone.enc.named_parameters():
                param.requires_grad = "lora_" in name
        return model

    def build(self, model):
        cfg = Config.fromfile(str(CONFIG))
        optimizer = build_optimizer(cfg.optimizer, model, cfg.param_dicts)
        cfg.scheduler.total_steps = 100
        return optimizer, build_scheduler(cfg.scheduler, optimizer)

    def test_block_selection_full_lora_and_ddp_prefix(self):
        for lora in (False, True):
            for wrapped in (False, True):
                with self.subTest(lora=lora, wrapped=wrapped):
                    model = self.make_model(lora)
                    expected_blocks = {
                        id(p) for module in model.backbone.modules()
                        if isinstance(module, Block)
                        for p in module.parameters() if p.requires_grad
                    }
                    self.assertTrue(expected_blocks)
                    main_params = list(model.head.parameters())
                    if not lora:
                        main_params += list(model.backbone.embedding.parameters())
                        main_params += list(model.backbone.enc.enc1.down.parameters())
                    if wrapped:
                        wrapper = nn.Module()
                        wrapper.module = model
                        model = wrapper
                    optimizer, _ = self.build(model)
                    main, block = optimizer.param_groups
                    main_ids = {id(p) for p in main["params"]}
                    block_ids = {id(p) for p in block["params"]}
                    self.assertEqual(block_ids, expected_blocks)
                    self.assertFalse(main_ids & block_ids)
                    self.assertTrue({id(p) for p in main_params} <= main_ids)
                    self.assertEqual(
                        main_ids | block_ids,
                        {id(p) for p in model.parameters() if p.requires_grad},
                    )

    def test_curve_and_resume(self):
        model = self.make_model()
        optimizer, scheduler = self.build(model)
        reference = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=1e-4)
        reference_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            reference, max_lr=1e-4, total_steps=100, pct_start=0.05,
            anneal_strategy="cos", div_factor=10.0, final_div_factor=10.0,
        )
        for step in range(100):
            main_lr, block_lr = scheduler.get_last_lr()
            self.assertAlmostEqual(main_lr, reference_scheduler.get_last_lr()[0], places=15)
            self.assertAlmostEqual(block_lr, main_lr * 0.1, places=15)
            if step in (0, 4, 99):
                expected = {0: 1e-5, 4: 1e-4, 99: 1e-6}[step]
                self.assertAlmostEqual(main_lr, expected, places=15)
            if step == 37:
                # Resume with fresh objects, as CheckpointLoader does.
                optimizer_state = copy.deepcopy(optimizer.state_dict())
                scheduler_state = copy.deepcopy(scheduler.state_dict())
                optimizer, scheduler = self.build(model)
                optimizer.load_state_dict(optimizer_state)
                scheduler.load_state_dict(scheduler_state)
            for param in model.parameters():
                if param.requires_grad:
                    param.grad = torch.ones_like(param)
            optimizer.step()
            scheduler.step()
            reference.step()
            reference_scheduler.step()

    def test_legacy_explicit_peaks(self):
        cfg = Config.fromfile(str(CONFIG))
        cfg.param_dicts = [ConfigDict(keyword="backbone.enc", lr=2e-4)]
        cfg.scheduler.max_lr = [1e-4, 2e-4]
        cfg.scheduler.total_steps = 100
        optimizer = build_optimizer(cfg.optimizer, self.make_model(), cfg.param_dicts)
        scheduler = build_scheduler(cfg.scheduler, optimizer)
        for _ in range(100):
            main_lr, encoder_lr = scheduler.get_last_lr()
            self.assertAlmostEqual(encoder_lr, main_lr * 2, places=15)
            optimizer.step()
            scheduler.step()


if __name__ == "__main__":
    unittest.main()
