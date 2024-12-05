import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import time

def prepare_dataset(csv_path):
    """
    Prepares the dataset from cleaned CSV file.
    """
    print(f"Reading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print("Columns found:", df.columns.tolist())
    
    dataset_dict = {
        "prompt": [],
        "completion": []
    }
    
    for idx, row in df.iterrows():
        try:
            # Create detailed clinical prompt
            prompt = (
                f"Patient Case {idx + 1}: "
                f"{row['DiseaseName']} in a {row['Age of Onset']} {row['Sex']} patient. "
                f"Clinical Features: {row['Clinical Manifestations']}. "
                f"Family History: {row['Family History']}. "
            )
            if pd.notna(row['Genetic Mutation']):
                prompt += f"Genetic Analysis: {row['Genetic Mutation']}. "
            
            # Create structured treatment and prognosis completion
            completion = (
                f"Treatment Plan: {row['Treatment options']}. "
                f"Prognosis: {row['Prognosis']}. "
            )
            if pd.notna(row['Research Studies']):
                completion += f"Research Context: {row['Research Studies']}"
            
            dataset_dict["prompt"].append(prompt)
            dataset_dict["completion"].append(completion)
            
        except Exception as e:
            print(f"Warning: Skipping row {idx} due to error: {e}")
            continue
    
    print(f"\nSuccessfully created {len(dataset_dict['prompt'])} prompt-completion pairs")
    return Dataset.from_dict(dataset_dict)

if __name__ == "__main__":
    # Test dataset preparation
    dataset = prepare_dataset("cleaned_dataset.csv")
    
    # Display some examples
    print("\nExample prompt-completion pairs:")
    for i in range(min(3, len(dataset))):
        print(f"\nExample {i+1}:")
        print("Prompt:", dataset[i]['prompt'])
        print("Completion:", dataset[i]['completion'])
        print("-" * 80)