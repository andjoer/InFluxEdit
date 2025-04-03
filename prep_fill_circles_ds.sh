#!/bin/bash

# Define the target directory
DATASET_DIR="data/my_fill50k"
DATA_DIR="data"
ZIP_FILE="$DATA_DIR/my_fill50k.zip"
DOWNLOAD_URL="https://www.cs.cmu.edu/~img2img-turbo/data/my_fill50k.zip"

# Check if the final dataset directory already exists
if [ -d "$DATASET_DIR" ]; then
    echo "Dataset directory '$DATASET_DIR' already exists. Skipping download and extraction."
else
    # If the directory doesn't exist, proceed with download and extraction
    echo "Dataset directory '$DATASET_DIR' not found. Proceeding with download..."

    mkdir -p "$DATA_DIR"
    # Check if the zip file exists from a partial download, remove it if so
    if [ -f "$ZIP_FILE" ]; then
        echo "Removing existing zip file '$ZIP_FILE' before download."
        rm "$ZIP_FILE"
    fi

    wget "$DOWNLOAD_URL" -O "$ZIP_FILE"

    # Check if wget was successful
    if [ $? -ne 0 ]; then
        echo "Error downloading the dataset. Please check the URL or your network connection."
        exit 1 # Exit on download error
    fi

    cd "$DATA_DIR"
    unzip my_fill50k.zip

    # Check if unzip was successful
    if [ $? -ne 0 ]; then
        echo "Error unzipping the file. It might be corrupted or incomplete."
        cd .. 
        exit 1 # Exit on unzip error
    fi

    rm my_fill50k.zip

    echo "Dataset downloaded and extracted successfully to '$DATASET_DIR'."

    cd ..
fi 

# Now call the python scripts 
echo "Combining training images..."
python dataset_tools/combine_images.py --dir_a "$DATASET_DIR/train_A" --dir_b "$DATASET_DIR/train_B" --output_dir "$DATASET_DIR/train_combined"

echo "Combining test images..."
python dataset_tools/combine_images.py --dir_a "$DATASET_DIR/test_A" --dir_b "$DATASET_DIR/test_B" --output_dir "$DATASET_DIR/test_combined"

echo "Extracting prompts..."
python dataset_tools/extract_prompts_from_json.py "$DATASET_DIR/train_prompts.json" --output_dir "$DATASET_DIR/prompts"

echo "Extracting prompts for combined images..."
python dataset_tools/extract_prompts_from_json.py "$DATASET_DIR/train_prompts.json" --output_dir "$DATASET_DIR/prompts_combined" --prefix "Before and after image. On the left half a cricle and on the right side the very same circle like this: "

echo "Dataset preparation complete."
exit 0