import torch
import numpy as np
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import matplotlib.pyplot as plt
import os
import datetime
import time
import shutil
import psutil
import math
import gc
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def print_gpu_memory(message: str):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        print(f"\r{message}: VRAM in GB Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}, allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f}, reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

class Trainer:
    def __init__(self, model, optimizer, device, batch_size, max_iters, eval_interval, eval_iters, learning_rate, weight_decay, warmup_iters, lr_decay_iters, min_lr, checkpoint_dir='checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.checkpoint_dir = checkpoint_dir
        self.best_loss = float('inf')
        
        # Training state
        self.current_iter = 0
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
        
        # Initialize training metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'grad_norm': [],
            'memory_usage': []
        }
        
        # Load data
        self.data = self.load_data()
        self.n = len(self.data)

        # Dynamically adjust batch size to fit 80% of GPU memory if using CUDA
        if self.device.startswith('cuda') and torch.cuda.is_available():
            self.batch_size = self._find_max_batch_size_within_vram(target_utilization=0.8)
            print(f"[Trainer] Adjusted batch size to {self.batch_size} to fit within 80% of GPU VRAM.")

    def load_data(self):
        """Load the training data"""
        try:
            data_file = os.path.join('src', 'data', 'train.bin')
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Training data file not found at {data_file}")
            
            # Load data as numpy array first
            data = np.memmap(data_file, dtype=np.uint16, mode='r')
            # Convert to tensor
            data = torch.from_numpy(data.copy())  # Make a copy to ensure it's writable
            return data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def get_batch(self, split):
        """Get a batch of data"""
        try:
            # Generate random indices
            ix = torch.randint(len(self.data) - self.model.config.block_size, (self.batch_size,))
            
            # Get input sequences
            x = torch.stack([self.data[i:i+self.model.config.block_size].long() for i in ix])
            # Get target sequences (shifted by 1)
            y = torch.stack([self.data[i+1:i+1+self.model.config.block_size].long() for i in ix])

            # Move to device
            x, y = x.to(self.device), y.to(self.device)
            
            # print_gpu_memory(f"After Move to device")
            return x, y
        except Exception as e:
            print(f"Error in get_batch: {str(e)}")
            raise

    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def estimate_loss(self):
        """Estimate loss on validation set"""
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                print_gpu_memory(f"{k} ./. {self.eval_iters}")
                try:
                    X, Y = self.get_batch(split)
                    with torch.no_grad():
                        with torch.amp.autocast('cuda', enabled=(self.device == 'cuda')):
                            logits, loss = self.model(X, Y)
                    losses[k] = loss.item()
                except Exception as e:
                    print(f"Error during evaluation: {str(e)}")
                    continue
            out[split] = losses.mean()
        self.model.train()
        return out

    def check_disk_space(self, required_space_mb=1000):
        """Check if there's enough disk space for saving the model"""
        try:
            # Get disk usage statistics
            disk_usage = psutil.disk_usage('/')
            free_space_mb = disk_usage.free / (1024 * 1024)  # Convert to MB
            
            if free_space_mb < required_space_mb:
                print(f"Warning: Low disk space. Only {free_space_mb:.2f}MB free, {required_space_mb}MB required")
                return False
            return True
        except Exception as e:
            print(f"Warning: Could not check disk space: {e}")
            return True  # Continue anyway if we can't check

    def save_checkpoint(self, iter_num, loss, is_best=False):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_num': iter_num,
                'loss': loss,
                'config': self.model.config,
            }
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{iter_num}.pt')
            torch.save(checkpoint, checkpoint_path)
            
            if is_best:
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                torch.save(checkpoint, best_path)
                print(f"Saved best model with loss {loss:.4f}")
            
            print(f"Saved checkpoint to {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            return False

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint with error handling"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.current_iter = checkpoint['iter_num']
            self.best_loss = checkpoint['loss']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.learning_rates = checkpoint['learning_rates']
            self.metrics = checkpoint.get('metrics', self.metrics)
            print(f"Successfully loaded checkpoint from iteration {self.current_iter}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def _find_max_batch_size_within_vram(self, target_utilization=0.8, min_batch=1, max_batch=None):
        """Estimate the largest batch size that fits within target_utilization of GPU VRAM."""
        torch.cuda.empty_cache()
        total_mem = torch.cuda.get_device_properties(0).total_memory
        max_mem = int(total_mem * target_utilization)
        # Use initial batch size as upper bound if not specified
        if max_batch is None:
            max_batch = self.batch_size
        block_size = getattr(self.model.config, 'block_size', 128)
        dtype_size = 2  # Assume fp16 (float16) for mixed precision
        # Try decreasing batch size until it fits
        for batch in range(max_batch, min_batch - 1, -1):
            try:
                dummy_x = torch.zeros((batch, block_size), dtype=torch.long, device=self.device)
                dummy_y = torch.zeros((batch, block_size), dtype=torch.long, device=self.device)
                with torch.amp.autocast("cuda", enabled=True):
                    _ = self.model(dummy_x, dummy_y)
                used = torch.cuda.max_memory_allocated(0)
                torch.cuda.empty_cache()
                if used < max_mem:
                    return batch
            except RuntimeError:
                torch.cuda.empty_cache()
                continue
        print("[Trainer] Warning: Could not fit even the minimum batch size in GPU memory. Using min_batch.")
        return min_batch

    def train(self):
        """Train the model"""
        print(f"Training started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        start_time = time.time()
        
        try:
            # Initialize training
            X, Y = self.get_batch('train')
            best_loss = float('inf')
            current_loss = None
            
            for iter_num in range(self.current_iter, self.max_iters):
                self.current_iter = iter_num
                print_gpu_memory(f"{self.current_iter} .. {self.max_iters}")
                # Determine and set the learning rate for this iteration
                lr = self.get_lr(iter_num)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.learning_rates.append(lr)
                
                # Forward pass with mixed precision
                #with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                with torch.amp.autocast('cuda', enabled=(self.device.startswith('cuda'))):
                    logits, loss = self.model(X, Y)
                current_loss = loss.item()
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Track training loss
                self.train_losses.append(current_loss)
                
                # Evaluate the model
                if iter_num % self.eval_interval == 0:
                    losses = self.estimate_loss()
                    print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                    self.val_losses.append(losses['val'])
                    
                    # Save best model
                    if losses['val'] < best_loss:
                        best_loss = losses['val']
                        self.save_checkpoint(iter_num, best_loss, is_best=True)
                
                # Get next batch
                X, Y = self.get_batch('train')
            
            # Save final checkpoint
            if current_loss is not None:
                self.save_checkpoint(self.max_iters, current_loss)
            
            # Calculate and print training duration
            end_time = time.time()
            duration = end_time - start_time
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            print(f"\nTraining completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total training time: {hours}h {minutes}m {seconds}s")
            
            return os.path.join(self.checkpoint_dir, 'best_model.pt')
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            print("Attempting to save checkpoint...")
            if current_loss is not None:
                self.save_checkpoint(iter_num, current_loss)
            raise

    def plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, 'g', label='train_loss')
        plt.plot(val_losses, 'r', label='validation_loss')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('training_losses.png')
        plt.close()

    def plot_metrics(self):
        """Plot training metrics"""
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(np.linspace(0, len(self.train_losses), len(self.val_losses)), self.val_losses, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot learning rate
        plt.subplot(1, 2, 2)
        plt.plot(self.learning_rates)
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'training_metrics.png'))
        plt.close()
