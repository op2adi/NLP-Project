#!/bin/bash
# This script trains a GAT model for sentiment graph parsing

# Parameters
DATASET=${1:-"norec"}  # Default to norec if not specified
EMBEDDING_ID=${2:-"32"}  # Default to 32 if not specified
GAT_HEADS=${3:-8}       # Default to 8 attention heads
GAT_DROPOUT=${4:-0.15}  # Default to 0.15 dropout rate
BATCH_SIZE=${5:-50}     # Default batch size
EPOCHS=${6:-100}        # Default number of epochs

# Step 1: Check if sentiment_graphs directory exists and create it if needed
if [ ! -d "sentiment_graphs" ]; then
    mkdir -p sentiment_graphs
fi

# Step 2: Check if the CoNLL-U files for the dataset exist
if [ ! -d "sentiment_graphs/$DATASET" ] || [ ! -f "sentiment_graphs/$DATASET/train.conllu" ]; then
    echo "CoNLL-U files not found. Converting JSON files to CoNLL-U format..."
    
    # Convert the dataset JSON files to CoNLL-U format
    python convert_to_conllu.py --json_dir ../../data/$DATASET --out_dir sentiment_graphs/$DATASET --setup head_final
    
    # Check if conversion was successful
    if [ ! -f "sentiment_graphs/$DATASET/head_final/train.conllu" ]; then
        echo "Error: Failed to convert JSON files to CoNLL-U format."
        exit 1
    else
        # Creating symbolic links to simplify file access
        mkdir -p sentiment_graphs/$DATASET
        ln -sf head_final/train.conllu sentiment_graphs/$DATASET/train.conllu
        ln -sf head_final/dev.conllu sentiment_graphs/$DATASET/dev.conllu
        ln -sf head_final/test.conllu sentiment_graphs/$DATASET/test.conllu
        echo "Successfully converted JSON files to CoNLL-U format."
    fi
fi

# Create output directories
MODEL_DIR="gat"
mkdir -p experiments/$DATASET/${MODEL_DIR}
mkdir -p logs/$DATASET/${MODEL_DIR}

# Display experiment information
echo "Training GAT model on $DATASET"
echo "--------------------------------------"
echo "Embedding ID: $EMBEDDING_ID"
echo "GAT Heads: $GAT_HEADS"
echo "GAT Dropout: $GAT_DROPOUT"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "--------------------------------------"

# Check if PyTorch Geometric is installed (required for GAT)
pip list | grep -q torch-geometric
if [ $? -ne 0 ]; then
    echo "Installing PyTorch Geometric for GAT support..."
    pip install torch-geometric
fi

# Prepare command for training
train_cmd="python src/main.py --config configs/sgraph.cfg --bridge gat \
     --gat_heads $GAT_HEADS --gat_dropout $GAT_DROPOUT \
     --train sentiment_graphs/$DATASET/train.conllu \
     --val sentiment_graphs/$DATASET/dev.conllu \
     --external embeddings/${EMBEDDING_ID}.zip \
     --batch_size $BATCH_SIZE \
     --epochs $EPOCHS \
     --dir experiments/$DATASET/${MODEL_DIR}/"

# Run the training command
echo "Running training: $train_cmd"
eval $train_cmd

echo "Training completed. Results are in experiments/$DATASET/${MODEL_DIR}/"