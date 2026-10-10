"""DDP diagnostic: normal training, paired same-input sync/no_sync backward,
then isolated all-reduce. no_sync is diagnostic only; no updates are applied in
paired trials, so unsynchronized gradients never become training checkpoints.
"""
import argparse
from contextlib import nullcontext
from functools import partial
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from benchmark_human_pose import TimedDataset, timed_collate, init_worker
from pointcept.utils.config import Config
from pointcept.utils.env import set_seed
from pointcept.models import build_model
from pointcept.datasets import build_dataset
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--pairs',type=int,default=30)
    args=p.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local);torch.set_num_threads(4);dist.init_process_group('nccl');set_seed(20261010)
    cfg=Config.fromfile('configs/human_pose/hmr-utonia-v1m1-0-bedlam.py')
    cfg.model.use_lora=False;cfg.model.backbone.freeze_encoder=False
    base=build_model(cfg.model).cuda().train()
    model=torch.nn.parallel.DistributedDataParallel(base,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    model._set_ddp_runtime_logging_sample_rate(1)
    optimizer=build_optimizer(cfg.optimizer,model,cfg.param_dicts)
    scaler=torch.amp.GradScaler('cuda')
    data=TimedDataset(build_dataset(cfg.data.train));sampler=DistributedSampler(data,num_replicas=world,rank=rank,seed=20261010)
    loader=DataLoader(data,batch_size=64,sampler=sampler,num_workers=4,collate_fn=timed_collate,pin_memory=True,drop_last=True,persistent_workers=True,worker_init_fn=partial(init_worker,seed=20261010+rank*4),generator=torch.Generator().manual_seed(20261010+rank))
    cfg.scheduler.total_steps=len(loader)*cfg.epoch;scheduler=build_scheduler(cfg.scheduler,optimizer)
    iterator=iter(loader)
    baseline=[];ddp_stats=[];paired=[]
    gradient_bytes=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    peer_access=[[torch.cuda.can_device_access_peer(i,j) if i!=j else True for j in range(world)] for i in range(world)]
    def get_batch():
        b=next(iterator);b.pop('_worker_prep_ms')
        return {k:v.cuda(non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in b.items()}
    def events():return [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    if rank==0:print('BASELINE_START',gradient_bytes,flush=True)
    for step in range(args.warmup+args.steps+1):
        t=time.perf_counter();batch=get_batch();optimizer.zero_grad();e=events();e[0].record()
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):out=model(batch);loss=out['loss']
        e[1].record()
        # DDP records the preceding backward's statistics at this forward.
        if args.warmup<step<=args.warmup+args.steps:
            ddp_stats.append(dict(step=step-1,data=model._get_ddp_logging_data()))
        scaler.scale(loss).backward();e[2].record()
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_grad)
        scaler.step(optimizer);scale=scaler.get_scale();scaler.update();skip=scaler.get_scale()<scale
        if not skip:scheduler.step()
        e[3].record();torch.cuda.synchronize()
        row=dict(step=step,total_ms=(time.perf_counter()-t)*1000,forward_ms=e[0].elapsed_time(e[1]),backward_ms=e[1].elapsed_time(e[2]),optimizer_ms=e[2].elapsed_time(e[3]),skipped=skip,points=len(batch['coord']))
        if args.warmup<=step<args.warmup+args.steps:baseline.append(row)
        if rank==0 and (step+1)%10==0:print('BASELINE',step+1,json.dumps(row),flush=True)
        del out,loss,batch
    # Identical weights/input/random state within each pair. Zero grads every
    # time; no optimizer updates in either arm. Alternate order to reduce drift.
    if rank==0:print('PAIRS_START',flush=True)
    for pair in range(args.pairs+4):
        batch=get_batch();torch.cuda.synchronize()
        cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
        result={}
        for sync in ([True,False] if pair%2==0 else [False,True]):
            optimizer.zero_grad();torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
            dist.barrier();torch.cuda.synchronize()
            e=events();t=time.perf_counter()
            with nullcontext() if sync else model.no_sync():
                e[0].record()
                with torch.amp.autocast('cuda',dtype=torch.bfloat16):out=model(batch);loss=out['loss']
                e[1].record();scaler.scale(loss).backward();e[2].record()
            torch.cuda.synchronize()
            result['sync' if sync else 'no_sync']=dict(forward_ms=e[0].elapsed_time(e[1]),backward_ms=e[1].elapsed_time(e[2]),forward_backward_ms=e[0].elapsed_time(e[2]),wall_ms=(time.perf_counter()-t)*1000,loss=float(loss.detach()))
            del out,loss
        if pair>=4:paired.append(dict(pair=pair-4,points=len(batch['coord']),**result))
        if rank==0 and (pair+1)%5==0:print('PAIR',pair+1,json.dumps(result),flush=True)
        del batch
    optimizer.zero_grad();torch.cuda.synchronize()
    # Pure collective latency with no backward computation to overlap it.
    bandwidth=[]
    for byte_count in [25*1024**2,gradient_bytes]:
        tensor=torch.zeros(byte_count//4,device='cuda',dtype=torch.float32)
        for i in range(10):dist.all_reduce(tensor)
        torch.cuda.synchronize();dist.barrier();ev=events();ev[0].record()
        for i in range(30):dist.all_reduce(tensor)
        ev[1].record();torch.cuda.synchronize();ms=ev[0].elapsed_time(ev[1])/30
        bandwidth.append(dict(bytes=byte_count,mean_ms=ms,algorithm_gb_s=byte_count/(ms*1e6),ring_bus_gb_s=byte_count/(ms*1e6)*2*(world-1)/world))
        del tensor
    result=dict(rank=rank,baseline=baseline,ddp_stats=ddp_stats,paired=paired,collectives=bandwidth,peer_access=peer_access,ddp_config=model._get_ddp_logging_data())
    results=[None]*world if rank==0 else None
    dist.gather_object(result,results,dst=0)
    if rank==0:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(args=vars(args),gradient_bytes=gradient_bytes,world_size=world,batch_per_gpu=64,max_point_tokens=cfg.model.max_point_tokens,torch=torch.__version__,nccl_version=torch.cuda.nccl.version(),ranks=results),indent=2))
        print('DONE',str(path),flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
