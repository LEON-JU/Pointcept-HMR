"""Real BEDLAM training benchmark; no checkpoints, evaluation or remote logging.

Run directly for one GPU or with torchrun for DDP. Batch is PER GPU.
CUDA events split stream elapsed time without synchronizing between stages.
Only the end of each step is synchronized. Worker preparation overlaps training.
"""
from __future__ import annotations
import argparse
import functools
import json
import os
from pathlib import Path
import random
import subprocess
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from pointcept.utils.config import Config
from pointcept.utils.env import set_seed
from pointcept.models import build_model
from pointcept.datasets import build_dataset
from pointcept.datasets.human_pose import human_pose_collate_fn
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler


class TimedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        t = time.perf_counter()
        sample = self.dataset[index]
        sample['_prep_ms'] = (time.perf_counter() - t) * 1000
        return sample


def timed_collate(samples):
    prep = sum(s.pop('_prep_ms') for s in samples)
    t = time.perf_counter()
    batch = human_pose_collate_fn(samples)
    batch['_worker_prep_ms'] = prep + (time.perf_counter() - t) * 1000
    return batch


def init_worker(worker_id, seed):
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


class Monitor:
    def __init__(self):
        self.rows = []
        self.proc = subprocess.Popen([
            'nvidia-smi', '--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu',
            '--format=csv,noheader,nounits', '-lms', '200'], stdout=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()
    def read(self):
        for line in self.proc.stdout:
            try:
                values = [float(x.strip()) for x in line.split(',')]
                self.rows.append(dict(timestamp=time.time(), gpu=int(values[0]), utilization=values[1], memory_utilization=values[2], memory_mib=values[3], power_w=values[4], sm_clock_mhz=values[5], temperature_c=values[6]))
            except ValueError:
                pass
    def close(self):
        self.proc.terminate()
        self.thread.join(timeout=5)
        self.proc.wait(timeout=5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/human_pose/hmr-utonia-v1m1-0-bedlam.py')
    p.add_argument('--mode', choices=['full', 'lora'], required=True)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--warmup', type=int, default=15)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261010)
    p.add_argument('--max-point-tokens', type=int)
    p.add_argument('--detail-stages', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    set_seed(args.seed)
    cfg = Config.fromfile(args.config)
    cfg.model.use_lora = args.mode == 'lora'
    cfg.model.backbone.freeze_encoder = False
    if args.max_point_tokens is not None:
        cfg.model.max_point_tokens = args.max_point_tokens
    if cfg.model.use_lora:
        # Adapter initialization must not change the common head initialization.
        import peft
        get_peft_model = peft.get_peft_model
        def isolated_adapter_init(*a, **kw):
            with torch.random.fork_rng(devices=[]):
                return get_peft_model(*a, **kw)
        peft.get_peft_model = isolated_adapter_init
    model = build_model(cfg.model).cuda().train()
    counts = dict(total=sum(p.numel() for p in model.parameters()), trainable=sum(p.numel() for p in model.parameters() if p.requires_grad), backbone_trainable=sum(p.numel() for p in model.backbone.parameters() if p.requires_grad), lora_trainable=sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and 'lora_' in n))
    if args.mode == 'lora':
        assert all(('lora_' in n) == p.requires_grad for n,p in model.backbone.named_parameters())
    else:
        assert all(p.requires_grad for p in model.backbone.parameters())
    original = model
    # Boundaries around the backbone, including its feature upsampling.
    timing = {}
    original_deep = original._deep_point
    def deep(data):
        timing['backbone_fwd_start'].record()
        point = original_deep(data)
        timing['backbone_fwd_end'].record()
        if point.feat.requires_grad:
            def grad_boundary(grad):
                timing['backbone_bwd_start'].record()
                return grad
            point.feat.register_hook(grad_boundary)
        return point
    original._deep_point = deep
    detail_events = {}
    if args.detail_stages:
        def event_pair(label):
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            detail_events.setdefault(label, []).append(pair)
            return pair
        def wrap_forward(obj, attr, label):
            fn = getattr(obj, attr)
            def wrapped(*a, **kw):
                begin, end = event_pair(label + '_forward_ms')
                begin.record()
                result = fn(*a, **kw)
                end.record()
                return result
            setattr(obj, attr, wrapped)
        def wrap_module(module, label):
            wrap_forward(module, 'forward', label)
            pending = []
            def before_backward(_module, _grad):
                pair = event_pair(label + '_backward_ms')
                pending.append(pair)
                pair[0].record()
            def after_backward(_module, _inputs, _outputs):
                pending.pop()[1].record()
            module.register_full_backward_pre_hook(before_backward)
            module.register_full_backward_hook(after_backward)
        wrap_forward(original, '_memory', 'memory_tokens')
        wrap_forward(original, '_smpl_forward', 'smpl_prediction')
        wrap_forward(original, '_loss', 'loss_including_gt_smpl')
        wrap_module(original.global_fusion, 'global_fusion')
        wrap_module(original.token_proj, 'token_projection')
        for layer in original.decoder:
            wrap_module(layer, 'decoder')
        for name in ['joint_seed', 'joint_up_proj', 'pose_head', 'joint_token_reduce', 'trans_head', 'beta_pool_mlp', 'beta_head']:
            wrap_module(getattr(original, name), 'pose_heads_and_queries')
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=cfg.find_unused_parameters)
    optimizer = build_optimizer(cfg.optimizer, model, cfg.param_dicts)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.enable_amp)
    dataset = TimedDataset(build_dataset(cfg.data.train))
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, num_workers=args.workers, collate_fn=timed_collate, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0, worker_init_fn=functools.partial(init_worker, seed=args.seed + rank * args.workers), generator=torch.Generator().manual_seed(args.seed + rank))
    cfg.scheduler.total_steps = len(loader) * cfg.epoch
    scheduler = build_scheduler(cfg.scheduler, optimizer)
    iterator = iter(loader)
    monitor = Monitor() if rank == 0 else None
    rows = []
    windows = []
    total = args.warmup + args.rounds * args.steps
    if rank == 0:
        print('BENCH_START', json.dumps(dict(**vars(args), world_size=world, counts=counts)), flush=True)
    for step in range(total):
        if step == args.warmup:
            torch.cuda.synchronize()
            if world > 1: dist.barrier()
            torch.cuda.reset_peak_memory_stats()
            window_start = time.time()
        timing = {name: torch.cuda.Event(enable_timing=True) for name in ['copy_start','copy_end','fwd_start','fwd_end','bwd_end','optim_end','backbone_fwd_start','backbone_fwd_end','backbone_bwd_start']}
        detail_events = {}
        start = time.perf_counter()
        batch = next(iterator)
        data_end = time.perf_counter()
        worker_prep = batch.pop('_worker_prep_ms')
        points = len(batch['coord'])
        timing['copy_start'].record()
        batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
        timing['copy_end'].record()
        optimizer.zero_grad()
        timing['fwd_start'].record()
        with torch.amp.autocast('cuda', enabled=cfg.enable_amp, dtype=getattr(torch, cfg.amp_dtype)):
            output = model(batch)
            loss = output['loss']
        timing['fwd_end'].record()
        scaler.scale(loss).backward()
        timing['bwd_end'].record()
        scaler.unscale_(optimizer)
        if cfg.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        scaler.step(optimizer)
        old_scale = scaler.get_scale()
        scaler.update()
        skipped = scaler.get_scale() < old_scale
        if not skipped: scheduler.step()
        timing['optim_end'].record()
        torch.cuda.synchronize()
        end = time.perf_counter()
        def elapsed(a,b): return timing[a].elapsed_time(timing[b])
        row = dict(step=step, round=(step-args.warmup)//args.steps, total_ms=(end-start)*1000, data_wait_ms=(data_end-start)*1000, worker_prep_ms=worker_prep, h2d_ms=elapsed('copy_start','copy_end'), zero_ms=elapsed('copy_end','fwd_start'), forward_ms=elapsed('fwd_start','fwd_end'), backward_ms=elapsed('fwd_end','bwd_end'), optimizer_ms=elapsed('bwd_end','optim_end'), backbone_forward_ms=elapsed('backbone_fwd_start','backbone_fwd_end'), backbone_backward_ms=elapsed('backbone_bwd_start','bwd_end'), points=points, loss=float(loss.detach()), skipped=skipped, allocated_gib=torch.cuda.memory_allocated()/2**30, reserved_gib=torch.cuda.memory_reserved()/2**30)
        row['head_forward_ms'] = row['forward_ms']-row['backbone_forward_ms']
        row['head_backward_ms'] = row['backward_ms']-row['backbone_backward_ms']
        row['other_ms'] = row['total_ms'] - sum(row[k] for k in ['data_wait_ms','h2d_ms','zero_ms','forward_ms','backward_ms','optimizer_ms'])
        for label, pairs in detail_events.items():
            row['detail_' + label] = sum(begin.elapsed_time(end) for begin, end in pairs)
        if not np.isfinite(row['loss']): raise RuntimeError('Nonfinite loss')
        if step >= args.warmup: rows.append(row)
        if rank == 0 and (step < 3 or (step+1)%5==0):
            print('STEP', step+1, 'warmup' if step < args.warmup else 'measure', json.dumps({k: round(row[k],3) for k in ['total_ms','data_wait_ms','forward_ms','backward_ms','optimizer_ms','allocated_gib','loss','skipped']}), flush=True)
        del output, loss, batch
    windows.append([window_start, time.time()])
    result = dict(rank=rank, rows=rows, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, windows=windows)
    if world > 1:
        gathered = [None]*world if rank==0 else None
        dist.gather_object(result, gathered, dst=0)
    else:
        gathered = [result]
    if rank == 0:
        monitor.close()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(dict(args=vars(args), world_size=world, counts=counts, device=torch.cuda.get_device_name(), torch=torch.__version__, ranks=gathered, gpu_samples=monitor.rows, config_text=Path(args.config).read_text()), indent=2))
        print('BENCH_DONE',str(output_path), 'mean_ms',np.mean([r['total_ms'] for r in rows]), 'peak_GiB',result['peak_allocated_gib'], flush=True)
    if world>1: dist.destroy_process_group()


if __name__ == '__main__':
    main()
