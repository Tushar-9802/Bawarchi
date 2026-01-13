# Manual Dataset Download Guide

This guide provides step-by-step instructions for manually downloading datasets that require special authentication or manual intervention.

## Table of Contents
1. [Kaggle API Setup](#kaggle-api-setup)
2. [Roboflow Ingredients Dataset](#roboflow-ingredients-dataset)
3. [Troubleshooting](#troubleshooting)

---

## Kaggle API Setup

Several datasets in this project are hosted on Kaggle and require API authentication.

### Step 1: Create Kaggle Account
1. Go to [https://www.kaggle.com](https://www.kaggle.com)
2. Sign up for a free account (if you don't have one)
3. Log in to your account

### Step 2: Generate API Token
1. Click on your profile picture (top right)
2. Go to **Account** → **API** section
3. Click **"Create New Token"**
4. This will download a file named `kaggle.json`

### Step 3: Install Kaggle API Token

#### Windows:
1. Press `Win + R` to open Run dialog
2. Type: `%USERPROFILE%\.kaggle` and press Enter
3. If the `.kaggle` folder doesn't exist, create it
4. Copy the downloaded `kaggle.json` file into this folder
5. The full path should be: `C:\Users\YOUR_USERNAME\.kaggle\kaggle.json`

#### Linux/Mac:
1. Open terminal
2. Create the directory:
   ```bash
   mkdir -p ~/.kaggle
   ```
3. Move the downloaded file:
   ```bash
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   ```
4. Set proper permissions (important for security):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Step 4: Verify Installation
Run this command in your terminal:
```bash
kaggle datasets list
```

If you see a list of datasets, the API is configured correctly!

### Step 5: Accept Dataset Licenses
Before downloading, you may need to accept dataset licenses:
1. Go to each dataset page on Kaggle
2. Click **"Download"** or **"New Notebook"** to accept the license
3. Required datasets:
   - [6000 Indian Food Recipes](https://www.kaggle.com/datasets/kanishk307/6000-indian-food-recipes-dataset)
   - [Indian Food 101](https://www.kaggle.com/datasets/nehaprabhavalkar/indian-food-101)
   - [Indian Food Images](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset)

---

## Roboflow Ingredients Dataset

The Food Ingredients Dataset requires manual download from Roboflow.

### Step 1: Create Roboflow Account
1. Go to [https://roboflow.com](https://roboflow.com)
2. Click **"Sign Up"** (free account available)
3. Verify your email address

### Step 2: Access the Dataset
1. Navigate to: [https://universe.roboflow.com/food-recipe-ingredient-images-0gnku/food-ingredients-dataset](https://universe.roboflow.com/food-recipe-ingredient-images-0gnku/food-ingredients-dataset)
2. Click **"Get Dataset"** or **"Download"** button
3. You may need to click **"Fork Dataset"** first if prompted

### Step 3: Export Dataset
1. On the dataset page, click **"Export"** or **"Download"**
2. Select export format:
   - **YOLOv8** (recommended for object detection)
   - **COCO** (alternative format)
   - **TensorFlow** (if using TensorFlow)
3. Click **"Continue"** or **"Download"**
4. Wait for the export to complete (may take a few minutes)

### Step 4: Download the Dataset
1. Once export is ready, click **"Download"**
2. Save the zip file to your Downloads folder
3. The file will be named something like: `food-ingredients-dataset-1.zip`

### Step 5: Extract the Dataset

#### Windows:
1. Right-click the downloaded zip file
2. Select **"Extract All..."**
3. Choose extraction location: `C:\Users\YOUR_USERNAME\Desktop\bawarchi\bawarchi_data\raw\ingredients`
   - Or navigate to where you're running the script and extract to: `./bawarchi_data/raw/ingredients`
4. Click **"Extract"**

#### Linux/Mac:
```bash
# Navigate to your project directory
cd /path/to/bawarchi

# Create the ingredients directory if it doesn't exist
mkdir -p bawarchi_data/raw/ingredients

# Extract the zip file
unzip ~/Downloads/food-ingredients-dataset-*.zip -d bawarchi_data/raw/ingredients
```

### Step 6: Verify Extraction
After extraction, your directory structure should look like:
```
bawarchi_data/
└── raw/
    └── ingredients/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/
        │   ├── images/
        │   └── labels/
        ├── test/
        │   ├── images/
        │   └── labels/
        └── data.yaml
```

### Step 7: Verify in Script
Run the dataset acquisition script again:
```bash
python scripts/dataset_acquisition.py
```

The script will detect the existing files and skip re-downloading.

---

## Alternative: Manual Dataset Downloads

If you prefer to download datasets manually without using APIs:

### RecipeNLG Dataset
- **Source**: HuggingFace
- **URL**: [https://huggingface.co/datasets/mbien/recipe_nlg](https://huggingface.co/datasets/mbien/recipe_nlg)
- **Method**: The script automatically downloads this via HuggingFace datasets library
- **No manual action needed**

### 6K Indian Recipes (Alternative)
- **Source**: Mendeley Data
- **URL**: [https://data.mendeley.com/datasets/xsphgmmh7b/1](https://data.mendeley.com/datasets/xsphgmmh7b/1)
- **Steps**:
  1. Click "Download All"
  2. Extract the zip file
  3. Find `IndianFoodDatasetCSV.csv`
  4. Copy to: `bawarchi_data/raw/recipes/IndianFoodDatasetCSV.csv`

### Indian Food 101 (Alternative)
- **Source**: Kaggle (requires account)
- **URL**: [https://www.kaggle.com/datasets/nehaprabhavalkar/indian-food-101](https://www.kaggle.com/datasets/nehaprabhavalkar/indian-food-101)
- **Steps**:
  1. Log in to Kaggle
  2. Go to the dataset page
  3. Click "Download" (accept license if prompted)
  4. Extract the zip file
  5. Find `indian_food.csv`
  6. Copy to: `bawarchi_data/raw/recipes/indian_food.csv`

### Khana Dataset
- **Source**: GitHub
- **URL**: [https://github.com/prabhuomkar/khana](https://github.com/prabhuomkar/khana)
- **Method**: The script automatically downloads labels from GitHub
- **Note**: Images are hosted separately and may require separate download if needed

---

## Troubleshooting

### Kaggle API Issues

**Error: "401 - Unauthorized"**
- Check that `kaggle.json` is in the correct location
- Verify the JSON file contains valid `username` and `key` fields
- Regenerate the token if expired

**Error: "403 - Forbidden"**
- Accept the dataset license on Kaggle website
- Make sure you're logged into the correct Kaggle account

**Error: "File not found"**
- The dataset structure may have changed
- Check the actual filenames in the downloaded dataset
- Update the script with correct filenames

### Roboflow Issues

**Export taking too long**
- Large datasets can take 10-30 minutes to export
- Keep the browser tab open
- Check your email for completion notification

**Download link expired**
- Export links expire after some time
- Re-export the dataset to get a new download link

**Extraction errors**
- Ensure you have enough disk space (datasets can be several GB)
- Try extracting with a different tool (7-Zip, WinRAR, etc.)

### General Issues

**Permission errors (Linux/Mac)**
```bash
# Fix permissions for kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Fix permissions for data directory
chmod -R 755 bawarchi_data/
```

**Disk space issues**
- Check available space: `df -h` (Linux/Mac) or check in File Explorer (Windows)
- Datasets can require 5-10 GB of space
- Clean up old downloads if needed

**Network timeouts**
- Use a stable internet connection
- Large downloads may take hours
- Consider using a download manager for large files

---

## Quick Reference: Directory Structure

After all downloads, your directory should look like:
```
bawarchi_data/
├── raw/
│   ├── recipes/
│   │   ├── recipenlg_full.parquet
│   │   ├── IndianFoodDatasetCSV.csv
│   │   └── indian_food.csv
│   ├── images/
│   │   ├── khana_labels.json
│   │   ├── khana_taxonomy.json
│   │   └── [indian-food-images-dataset files]
│   └── ingredients/
│       ├── train/
│       ├── valid/
│       ├── test/
│       └── data.yaml
└── processed/
    └── recipes/
        └── unified_recipes.parquet
```

---

## Need Help?

If you encounter issues not covered here:
1. Check the error messages in the script output
2. Verify all prerequisites are installed: `pip install -r requirements.txt`
3. Ensure you have internet connectivity
4. Check that all required accounts are created and verified

