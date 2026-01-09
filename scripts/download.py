from datasets import load_dataset
import pandas as pd
import os

# Download full dataset
dataset = load_dataset("mbien/recipe_nlg", split="train")

# Ensure output directory exists
output_dir = "data/recipes/raw"
os.makedirs(output_dir, exist_ok=True)

# Convert to pandas DataFrame and save as CSV
df = dataset.to_pandas()
df.to_csv(os.path.join(output_dir, "recipe_nlg.csv"), index=False)