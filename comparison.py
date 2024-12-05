import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
import time
import pandas as pd
import numpy as np

def prepare_data():
    """
    Reads and prepares the dataset, with detailed error handling
    """
    try:
        df = pd.read_csv("big_dataset.csv")
        print("Available columns:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        raise

def train_model(mode='serial', num_epochs=3):
    """
    Trains the model in either serial or parallel mode and tracks performance
    """
    print(f"\nInitializing {mode} training...")
    start_setup = time.time()
    
    # Initialize accelerator for distributed training
    accelerator = Accelerator() if mode == 'parallel' else None
    
    # Load model
    print("Loading GPT-2 model...")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare dataset
    print("Preparing dataset...")
    df = prepare_data()
    
    # Create simpler text representations for training
    # Using whatever columns are available in your dataset
    texts = []
    for _, row in df.iterrows():
        text = ""
        for col in df.columns:
            if pd.notna(row[col]):  # Only include non-null values
                text += f"{col}: {row[col]}\n"
        texts.append(text)
    
    print(f"Tokenizing {len(texts)} examples...")
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask']
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True
    )
    
    # Metrics tracking
    metrics = {
        'epoch_times': [],
        'losses': [],
        'mode': mode,
        'setup_time': time.time() - start_setup
    }
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    if mode == 'parallel':
        print("Preparing for parallel execution...")
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Training loop
    print(f"\nStarting {mode} training loop...")
    model.train()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_losses = []
        
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
            try:
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss
                
                # Record loss
                epoch_losses.append(loss.item())
                
                # Backward pass
                if mode == 'parallel':
                    accelerator.backward(loss)
                else:
                    loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                if batch_idx % 5 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}")
                    
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        epoch_time = time.time() - epoch_start
        metrics['epoch_times'].append(epoch_time)
        metrics['losses'].append(np.mean(epoch_losses))
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Time: {epoch_time:.2f}s")
        print(f"Average Loss: {metrics['losses'][-1]:.4f}")
    
    return metrics

if __name__ == "__main__":
    print("Starting performance comparison...")
    
    try:
        print("\nRunning serial training...")
        serial_metrics = train_model(mode='serial')
        
        print("\nRunning parallel training...")
        parallel_metrics = train_model(mode='parallel')
        
        # Performance analysis
        avg_serial_time = np.mean(serial_metrics['epoch_times'])
        avg_parallel_time = np.mean(parallel_metrics['epoch_times'])
        speedup = avg_serial_time / avg_parallel_time
        
        print("\n" + "="*50)
        print("Performance Summary:")
        print("="*50)
        print(f"Serial Training:")
        print(f"  - Setup Time: {serial_metrics['setup_time']:.2f}s")
        print(f"  - Average Epoch Time: {avg_serial_time:.2f}s")
        print(f"  - Final Loss: {serial_metrics['losses'][-1]:.4f}")
        print(f"\nParallel Training:")
        print(f"  - Setup Time: {parallel_metrics['setup_time']:.2f}s")
        print(f"  - Average Epoch Time: {avg_parallel_time:.2f}s")
        print(f"  - Final Loss: {parallel_metrics['losses'][-1]:.4f}")
        print(f"\nSpeedup Analysis:")
        print(f"  - Overall Speedup Factor: {speedup:.2f}x")
        print("="*50)
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise