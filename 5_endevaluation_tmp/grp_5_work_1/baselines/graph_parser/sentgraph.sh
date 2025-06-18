#!/bin/bash
# This script runs experiments with different bridge mechanisms for sentiment graph parsing

# Default settings
DATASET=${1:-"multibooked_ca"}
EMBEDDING_ID=${2:-"32"}
BRIDGE=${3:-"dpa+"}
GAT_HEADS=${4:-4}
GAT_DROPOUT=${5:-0.1}
BATCH_SIZE=${6:-50}
EPOCHS=${7:-100}

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
mkdir -p logs/$DATASET/${BRIDGE}
mkdir -p experiments/$DATASET/${BRIDGE}

# Display experiment information
echo "Running $DATASET with $BRIDGE bridge"
echo "--------------------------------------"
echo "Embedding ID: $EMBEDDING_ID"
echo "GAT Heads: $GAT_HEADS (only used with 'gat' bridge)"
echo "GAT Dropout: $GAT_DROPOUT (only used with 'gat' bridge)"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "--------------------------------------"

# Check if PyTorch Geometric is installed (required for GAT)
if [ "$BRIDGE" = "gat" ]; then
    pip list | grep -q torch-geometric
    if [ $? -ne 0 ]; then
        echo "Installing PyTorch Geometric for GAT support..."
        pip install torch-geometric
    fi
fi

# Prepare command
cmd="python src/main.py --config configs/sgraph.cfg --bridge $BRIDGE"

# Add GAT-specific parameters if using GAT bridge
if [ "$BRIDGE" = "gat" ]; then
    cmd="$cmd --gat_heads $GAT_HEADS --gat_dropout $GAT_DROPOUT"
fi

# Add remaining parameters
cmd="$cmd --train sentiment_graphs/$DATASET/train.conllu \
     --val sentiment_graphs/$DATASET/dev.conllu \
     --external embeddings/${EMBEDDING_ID}.zip \
     --batch_size $BATCH_SIZE \
     --epochs $EPOCHS \
     --dir experiments/$DATASET/${BRIDGE}/"

# Run the command
echo "Running: $cmd"
eval $cmd

echo "Experiment completed. Results are in experiments/$DATASET/${BRIDGE}/"
