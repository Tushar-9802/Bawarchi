from datasets import load_dataset
import pandas as pd
import os
import sys

# Ensure output directory exists
output_dir = "data/recipes/raw"
os.makedirs(output_dir, exist_ok=True)

# Check if manual data directory exists
manual_data_dir = os.path.join(output_dir, "manual_data")
manual_csv_path = os.path.join(manual_data_dir, "full_dataset.csv")

# Try to load dataset - first check if manual data exists
dataset = None
df = None

if os.path.exists(manual_csv_path):
    print(f"Found manual data at {manual_csv_path}. Loading dataset...")
    try:
        dataset_dict = load_dataset("mbien/recipe_nlg", data_dir=manual_data_dir, trust_remote_code=True)
        # Handle both DatasetDict and Dataset
        if hasattr(dataset_dict, "keys"):
            if "train" in dataset_dict:
                dataset = dataset_dict["train"]
            else:
                # If no train split, use the first available split
                dataset = list(dataset_dict.values())[0]
        else:
            dataset = dataset_dict
    except Exception as e:
        print(f"Error loading from manual data: {e}")
        print("Trying alternative loading method - reading CSV directly...")
        # Try loading CSV directly
        try:
            df = pd.read_csv(manual_csv_path)
            print(f"Successfully loaded CSV with {len(df)} rows.")
        except Exception as e2:
            print(f"Error reading CSV directly: {e2}")
            sys.exit(1)
elif os.path.exists(manual_data_dir):
    print(f"Manual data directory exists but full_dataset.csv not found at {manual_csv_path}")
    print("Please ensure full_dataset.csv is in the manual_data directory.")
    sys.exit(1)
else:
    print("Manual data not found. Attempting to download from HuggingFace...")
    try:
        dataset_dict = load_dataset("mbien/recipe_nlg", split="train", trust_remote_code=True)
        # Handle both DatasetDict and Dataset
        if hasattr(dataset_dict, "keys"):
            if "train" in dataset_dict:
                dataset = dataset_dict["train"]
            else:
                dataset = list(dataset_dict.values())[0]
        else:
            dataset = dataset_dict
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR: This dataset requires manual download.")
        print("="*70)
        print("\nPlease follow these steps:")
        print("1. Go to https://recipenlg.cs.put.poznan.pl/")
        print("2. Download the dataset.zip file")
        print("3. Unzip the file and locate full_dataset.csv")
        print(f"4. Place full_dataset.csv in: {manual_data_dir}")
        print(f"\nThen run this script again.")
        print("="*70)
        sys.exit(1)

# Convert to pandas DataFrame if needed
if df is None:
    if dataset is not None:
        print("Converting dataset to pandas DataFrame...")
        df = dataset.to_pandas()
    else:
        print("ERROR: No dataset or CSV data available.")
        sys.exit(1)

output_csv = os.path.join(output_dir, "recipe_nlg.csv")
print(f"Saving dataset to {output_csv}...")
df.to_csv(output_csv, index=False)
print(f"Successfully saved {len(df)} recipes to {output_csv}")