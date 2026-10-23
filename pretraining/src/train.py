from data_module import *
from model import Transformer, ModelArgs
from optim import *
from checkpoint import *

from pathlib import Path
from contextlib import nullcontext
import os
import json
from tqdm import tqdm

import torch 
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist


def main():
    rank, local_rank, world_size = int(os.environ.get('RANK', 0)), int(os.environ.get('LOCAL_RANK', 0)), int(os.environ.get('WORLD_SIZE', 0))
    is_ddp = world_size > 1
    if world_size == 0:
        world_size = 1
    
    if is_ddp:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ########
    
    global_batch_size =  512
    context_len = 1024
    num_epoch = 1
    max_lr = 3e-3
    min_lr = 3e-5
    warmup_ratio = 0.01
    weight_decay = 0.1
    max_l2_clip = 1.0
    
    micro_batch_size = 16
    
    resume_ckpt = None #ckpt_p / 'ckpt_3'
    
    seed = 20261022
    ########
    
    assert global_batch_size % (micro_batch_size * world_size) == 0
    gradient_accumulation_steps = global_batch_size // (micro_batch_size * world_size)
    
    current_dir = Path(__file__).resolve().parent.parent
    train_data_p = current_dir / 'dataset' / 'train.bin'
    valid_data_p = current_dir / 'dataset' / 'valid.bin'
    model_para_p = current_dir / 'src' / 'settings' / 'model_params.json'
    log_p = current_dir / 'log' / 'training_log.json'
    ckpt_p = current_dir / 'checkpoint'
    log_p.parent.mkdir(parents=True, exist_ok=True)
    ckpt_p.mkdir(parents=True, exist_ok=True) 
    
    train_tokenized = TokenizedDataset(train_data_p, context_len)
    train_sampler = DistributedSampler(dataset = train_tokenized,
                                num_replicas = world_size,
                                rank = rank,
                                shuffle = True,
                                seed = seed,
                                drop_last = False,
                                ) if is_ddp else None
    
    train_dataset = iter(torch.utils.data.DataLoader(
                    dataset = train_tokenized, 
                    batch_size = micro_batch_size,
                    sampler = train_sampler,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=2,
                    shuffle = not is_ddp,
                    drop_last = False ))
    
    valid_tokenized = TokenizedDataset(valid_data_p, context_len)
    valid_sampler = DistributedSampler(dataset = valid_tokenized,
                                num_replicas = world_size,
                                rank = rank,
                                shuffle = False,
                                drop_last = False,
                                ) if is_ddp else None
    
    valid_dataset = torch.utils.data.DataLoader(
                    dataset = valid_tokenized, 
                    batch_size = 8,
                    sampler = valid_sampler,
                    shuffle = False,
                    drop_last = False )
    
    total_step = len(train_dataset) * num_epoch // gradient_accumulation_steps
    warmup_step = int(warmup_ratio * total_step)
    
    valid_step = total_step // 20
    ckpt_step = total_step // 10
    log_step = 100
    
    with open(model_para_p, 'r') as f:
        model_args = ModelArgs(**json.load(f))
    
    model = Transformer(model_args).to(device)
    model.compile()
    
    if is_ddp:  model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    opt = AdamW(model.parameters(), lr=max_lr, betas = (0.9, 0.95), eps=1e-6, weight_decay = weight_decay)
    current_step = 0
    start = 0
    log = {'it':[], 'train_loss':[], 'valid_loss':[], 'lr': []}
    
    if resume_ckpt is not None:
        current_step, log = load_all(resume_ckpt, model, opt, log_p)
        start = current_step * gradient_accumulation_steps
        for _ in range(start):
            next(train_dataset)
    
    train_token, train_loss = 0, 0
    
    if is_ddp:
        train_sampler.set_epoch(epoch=0)
        valid_sampler.set_epoch(epoch=0)
    
    print(valid_step)
    
    for idx, batch in tqdm(enumerate(train_dataset, start=start), total=len(train_dataset), disable=(rank != 0)):
        x, y = batch['input_ids'].to(device).long(), batch['labels'].to(device).long()
        is_step = ((idx+1) % gradient_accumulation_steps == 0)
        sync_context = (model.no_sync() if is_ddp and not is_step else nullcontext())
        
        with sync_context:
            with torch.amp.autocast('cuda', dtype = torch.bfloat16):
                y_ = model(input_ids = x)
                loss = F.cross_entropy(y_.reshape(-1, y_.size(-1)), y.reshape(-1)) / gradient_accumulation_steps
    
            del y_ 
    
            train_token += y.numel()
            train_loss += (loss * y.numel() * gradient_accumulation_steps).detach()
            loss.backward()
            del loss
    
        if is_step:
            gradient_clipping(list(model.parameters()), max_l2_clip)
            dynamic_lr = get_lr_cosine_schedule(current_step, max_lr, min_lr, warmup_step, total_step) 
            for param_group in opt.param_groups:
                param_group['lr'] = dynamic_lr
            
            opt.step()
            opt.zero_grad()
            current_step += 1
    
            valid_loss_log = None
    
            if current_step % valid_step == 0:
    
                model.eval()
                valid_loss, valid_token = 0, 0
                for idx_val, batch_val in enumerate(valid_dataset):
                    x_val, y_val = batch_val['input_ids'].long().to(device), batch_val['labels'].long().to(device)
                    with torch.no_grad(), torch.amp.autocast('cuda', dtype = torch.bfloat16):
                        y_val_= model(input_ids = x_val)
                        loss_val = F.cross_entropy(y_val_.reshape(-1, y_val_.size(-1)), y_val.reshape(-1))
                        valid_token += y_val.numel()
                        valid_loss += (loss_val * y_val.numel()).detach()
                    del y_val_, loss_val, x_val, y_val
                valid_loss /= valid_token
                model.train()
    
                valid_loss_log = valid_loss.detach().clone()
                if is_ddp:
                    dist.all_reduce(valid_loss_log, op=dist.ReduceOp.SUM)
                    valid_loss_log /= world_size
                valid_loss_log = valid_loss_log.item()
    
            if current_step % log_step == 0 or valid_loss_log is not None:
                train_loss_log = train_loss.detach().clone()
                if is_ddp:
                    dist.all_reduce(train_loss_log, op=dist.ReduceOp.SUM)
                    train_loss_log /= world_size
                if rank == 0:
                    log['it'].append(current_step)
                    log['train_loss'].append(train_loss_log.item()/train_token)
                    log['valid_loss'].append(valid_loss_log)
                    log['lr'].append(dynamic_lr)
    
                    with open(log_p, "w") as f:
                        json.dump(log, f, indent=4)
    
                train_loss, train_token = 0, 0
            
            if current_step % ckpt_step == 0 and rank == 0:
                out = ckpt_p / ('ckpt_'+str(current_step))
                raw_model = model.module if is_ddp else model 
                save_checkpoint(raw_model, opt, current_step, out)
    
    if rank == 0:
        out = ckpt_p / ('ckpt_final')
        raw_model = model.module if is_ddp else model 
        save_checkpoint(raw_model, opt, current_step, out)
             
    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
