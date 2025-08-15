import torch
from data.data_processor import DataProcessor
from model.gpt import GPT, GPTConfig
from config import TrainingConfig
from training.trainer import Trainer, print_gpu_memory
import tiktoken
import os

def optimize_memory_for_t1200():
    """Optimize CUDA memory settings for 4GB T1200 GPU"""
    # Set memory allocation strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Clear cache and set memory fraction
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Reserve only 90% of GPU memory to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.9)


def clear_gpu_memory():
    """Clear GPU memory and show available memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print_gpu_memory("After torch.cuda.empty_cache()")


def main():
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Clear GPU memory first
    clear_gpu_memory()


    # Optimize memory settings first
    optimize_memory_for_t1200()

    torch.manual_seed(42)
    
    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("lora_checkpoints", exist_ok=True)
    
    # Initialize data processor and prepare dataset
    print("Preparing dataset...")
    data_processor = DataProcessor()
    data_processor.prepare_dataset()
    
    # Initialize model
    print("Initializing model...")
    model_config = GPTConfig()
    model = GPT(model_config)
    model = model.to(device)
    print_gpu_memory("After model = model.to(device)")
    
   

    print("Starting training...")
    training_config = TrainingConfig()
    #trainer = Trainer(model, training_config.batch_size, training_config.device)
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay),
        device=device,
        batch_size=training_config.batch_size,  
        max_iters=training_config.max_iters,
        eval_interval=training_config.eval_interval,
        eval_iters=training_config.eval_iters,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_iters=training_config.warmup_iters,
        lr_decay_iters=training_config.lr_decay_iters,
        min_lr=training_config.min_lr

    )

    # Train the model
    print("Training started...")
    print_gpu_memory("Before trainer.train()")
    print("This may take a while, please be patient...")
    best_model_path = trainer.train()
    
    # Load the best model for inference
    print("Loading best model for inference...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Example generation
    print("\nGenerating example story...")
    enc = tiktoken.get_encoding("gpt2")
    sentence = "Once upon a time there was a pumpkin."
    context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim=0).to(device)
    y = model.generate(context, 200)
    print(enc.decode(y.squeeze().tolist()))
    
    print("\nTraining completed! You can now run LoRA finetuning using:")
    print("python src/finetune_lora.py")

if __name__ == "__main__":
    main() 