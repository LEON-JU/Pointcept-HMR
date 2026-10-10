"""CPU regression checks for epoch/step evaluation and checkpoint positions."""
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from pointcept.engines.defaults import default_config_parser
from pointcept.engines.hooks.human_pose_evaluator import HumanPoseEvaluator
from pointcept.engines.hooks.misc import CheckpointSaver
from pointcept.engines.train import Trainer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/human_pose/hmr-utonia-v1m1-0-bedlam.py"


class ScheduleTests(unittest.TestCase):
    def test_config_and_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = default_config_parser(str(CONFIG), {"save_path": tmp})
            self.assertEqual((cfg.epoch, cfg.eval_every_step, cfg.data.train.loop), (2, 5000, 1))
            self.assertNotIn("eval_epoch", cfg)
            cfg = default_config_parser(str(CONFIG), {
                "save_path": tmp, "epoch": 3, "eval_every_step": 5000,
            })
            self.assertEqual((cfg.epoch, cfg.eval_every_step, cfg.data.train.loop), (3, 5000, 1))
            trainer = Trainer.__new__(Trainer)
            trainer.cfg = cfg
            trainer.train_loader = range(10)
            trainer.optimizer = torch.optim.AdamW([
                {"params": [torch.nn.Parameter(torch.zeros(1))]},
                {"params": [torch.nn.Parameter(torch.zeros(1))]},
            ])
            scheduler = trainer.build_scheduler()
            self.assertEqual(scheduler.total_steps, 30)
            for options in ({"eval_every_step": -1}, {"epoch": 0}, {"eval_epoch": 1}):
                with self.assertRaises(ValueError):
                    default_config_parser(str(CONFIG), {"save_path": tmp, **options})

    def make_trainer(self, path, interval):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        return SimpleNamespace(
            cfg=SimpleNamespace(evaluate=True, eval_every_step=interval,
                                save_path=path, enable_amp=False),
            model=model, optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 1),
            logger=logging.getLogger("schedule-test"), comm_info={"iter": 0},
            global_step=0, epoch=0, best_metric_value=-torch.inf,
        )

    def test_cadence_best_checkpoint_and_epoch_boundary(self):
        for interval, expected in ((0, [3, 6]), (2, [2, 3, 4, 6])):
            with self.subTest(interval=interval), tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "model").mkdir()
                trainer = self.make_trainer(tmp, interval)
                evaluator = HumanPoseEvaluator()
                evaluator.trainer = trainer
                evaluations = []

                def evaluate():
                    trainer.model.eval()
                    evaluations.append(trainer.global_step)
                    trainer.comm_info.update(current_metric_value=10 - trainer.global_step,
                                             current_metric_name="test_loss")

                evaluator.eval = evaluate
                saver = CheckpointSaver(metric_mode="min", save_epoch_freq=1, save_last_freq=2)
                saver.trainer = trainer
                saver.before_train()
                for epoch in range(2):
                    trainer.epoch = epoch
                    for iteration in range(3):
                        trainer.comm_info["iter"] = iteration
                        trainer.global_step += 1
                        evaluator.after_step()
                        saver.after_step()
                        # Unchanged optimizer step: accumulation or a skipped AMP update.
                        evaluator.after_step()
                        saver.after_step()
                        self.assertTrue(trainer.model.training)
                        if interval and trainer.global_step == 2:
                            best = torch.load(Path(tmp) / "model/model_best.pth", weights_only=False)
                            self.assertEqual((best["epoch"], best["iter"], best["global_step"]), (0, 2, 2))
                    evaluator.after_epoch()
                    saver.after_epoch()
                    checkpoint = torch.load(Path(tmp) / f"model/epoch_{epoch+1}.pth", weights_only=False)
                    self.assertEqual((checkpoint["epoch"], checkpoint["iter"]), (epoch+1, 0))
                self.assertEqual(evaluations, expected)
                self.assertEqual(trainer.best_metric_value, 4)

    def test_disabled_resume_and_exception_restore(self):
        trainer = self.make_trainer("/unused", 2)
        evaluator = HumanPoseEvaluator()
        evaluator.trainer = trainer
        evaluator.eval = Mock()
        trainer.cfg.evaluate = False
        trainer.global_step = 2
        evaluator.after_step()
        evaluator.after_epoch()
        evaluator.eval.assert_not_called()
        trainer.cfg.evaluate = True
        # Resume past step 2; the next interval is still global step 4.
        trainer.global_step = 3
        evaluator.after_step()
        evaluator.eval.assert_not_called()
        trainer.global_step = 4
        evaluator.after_step()
        evaluator.eval.assert_called_once()

        def fail():
            trainer.model.eval()
            raise RuntimeError("evaluation failed")

        evaluator.eval = fail
        trainer.global_step = 6
        with self.assertRaises(RuntimeError):
            evaluator.after_step()
        self.assertTrue(trainer.model.training)


if __name__ == "__main__":
    unittest.main()
