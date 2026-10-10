import torch
import torch.distributed as dist

import pointcept.utils.comm as comm
from pointcept.utils import swanlab
from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module("HumanPoseEvaluator")
class HumanPoseEvaluator(HookBase):
    """Distributed BEHAVE/BEDLAM evaluator using camera-coordinate MPJPE."""

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self._evaluate_once()

    def after_step(self):
        interval = self.trainer.cfg.eval_every_step
        step = self.trainer.global_step
        if self.trainer.cfg.evaluate and interval > 0 and step > 0 and step % interval == 0:
            self._evaluate_once()

    def _evaluate_once(self):
        step = self.trainer.global_step
        if getattr(self, "_last_eval_step", None) == step:
            return
        was_training = self.trainer.model.training
        self.trainer.logger.info("Evaluating at global optimizer step %d", step)
        try:
            self.eval()
        finally:
            self.trainer.model.train(was_training)
        self._last_eval_step = step
        self.trainer.comm_info["eval_step"] = step

    def eval(self):
        self.trainer.model.eval()
        mpjpe_sum = torch.zeros(1, device="cuda")
        smpl_joint_sum = torch.zeros(1, device="cuda")
        cr_vert_sum = torch.zeros(1, device="cuda")
        sample_count = torch.zeros(1, device="cuda")
        with torch.no_grad():
            for batch in self.trainer.val_loader:
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.cuda(non_blocking=True)
                output = self.trainer.model(batch)
                pred = output.get("pred_joints_cam")
                # Evaluation must use the same SMPL-X forward pass as the loss;
                # shard joints are retained only as input metadata.
                gt = output.get("gt_joints_cam")
                if pred is None or gt is None:
                    raise RuntimeError(
                        "HumanPoseEvaluator requires pred_joints_cam and gt_joints_cam; "
                        "check that SMPL-X assets are configured and evaluation batches include joints."
                    )
                if "loss_smpl_joint" not in output or "loss_cr_vert" not in output:
                    raise RuntimeError(
                        "HumanPoseEvaluator requires SMPL joint and vertex losses."
                    )
                n = min(pred.shape[1], gt.shape[1], 22)
                mpjpe_sum += torch.linalg.norm(pred[:, :n] - gt[:, :n],
                                               dim=-1).mean(1).sum()
                batch_size = pred.shape[0]
                smpl_joint_sum += output["loss_smpl_joint"] * batch_size
                cr_vert_sum += output["loss_cr_vert"] * batch_size
                sample_count += batch_size
        if comm.get_world_size() > 1:
            dist.all_reduce(mpjpe_sum)
            dist.all_reduce(smpl_joint_sum)
            dist.all_reduce(cr_vert_sum)
            dist.all_reduce(sample_count)
        value = (mpjpe_sum / sample_count.clamp_min(1)).item()
        smpl_joint = (smpl_joint_sum / sample_count.clamp_min(1)).item()
        cr_vert = (cr_vert_sum / sample_count.clamp_min(1)).item()
        eval_loss = smpl_joint + cr_vert
        self.trainer.storage.put_scalar("val/mpjpe", value)
        self.trainer.storage.put_scalar("val/smpl_joint", smpl_joint)
        self.trainer.storage.put_scalar("val/cr_vert", cr_vert)
        self.trainer.storage.put_scalar("val/loss", eval_loss)
        if self.trainer.writer is not None:
            for name, metric in (("mpjpe", value), ("smpl_joint", smpl_joint),
                                 ("cr_vert", cr_vert), ("loss", eval_loss)):
                self.trainer.writer.add_scalar(
                    "val/" + name, metric, self.trainer.global_step)
        if self.trainer.writer is not None and self.trainer.cfg.enable_swanlab:
            swanlab.log(
                {
                    "Epoch": self.trainer.epoch + 1,
                    "val/mpjpe": value,
                    "val/smpl_joint": smpl_joint,
                    "val/cr_vert": cr_vert,
                    "val/loss": eval_loss,
                },
                step=self.trainer.global_step,
            )
        self.trainer.comm_info["current_metric_value"] = eval_loss
        self.trainer.comm_info["current_metric_name"] = "val/smpl_joint+cr_vert"
        self.trainer.logger.info("Val result: MPJPE {:.6f} m".format(value))
        self.trainer.logger.info(
            "Val loss: smpl_joint {:.6f} + cr_vert {:.6f} = {:.6f}".format(
                smpl_joint, cr_vert, eval_loss
            )
        )
