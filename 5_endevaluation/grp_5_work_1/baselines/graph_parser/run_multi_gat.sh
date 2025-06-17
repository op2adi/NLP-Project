#!/bin/bash
# This script runs experiments with our enhanced multi-attention GAT model for sentiment graph parsing

# Default settings
DATASET=${1:-"multibooked_ca"}
EMBEDDING_ID=${2:-"32"}
GAT_HEADS=${3:-8}
GAT_DROPOUT=${4:-0.15}
BATCH_SIZE=${5:-50}
EPOCHS=${6:-100}

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
MODEL_DIR="multi_attention_gat"
mkdir -p logs/$DATASET/${MODEL_DIR}
mkdir -p experiments/$DATASET/${MODEL_DIR}

# Display experiment information
echo "Running $DATASET with enhanced Multi-Attention GAT bridge"
echo "--------------------------------------"
echo "Embedding ID: $EMBEDDING_ID"
echo "GAT Heads: $GAT_HEADS"
echo "GAT Dropout: $GAT_DROPOUT"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "--------------------------------------"

# Check if PyTorch Geometric is installed (required for GAT components)
pip list | grep -q torch-geometric
if [ $? -ne 0 ]; then
    echo "Installing PyTorch Geometric..."
    pip install torch-geometric
fi

# Prepare command for training
train_cmd="python src/main.py --config configs/sgraph.cfg --bridge multi_attention_gat \
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

# Evaluate on test set using the best model
echo "--------------------------------------"
echo "Evaluating best model on test set"
echo "--------------------------------------"

# Run prediction on test data using the best model
test_cmd="python src/main.py --config configs/sgraph.cfg --bridge multi_attention_gat \
     --gat_heads $GAT_HEADS --gat_dropout $GAT_DROPOUT \
     --predict_file sentiment_graphs/$DATASET/test.conllu \
     --external embeddings/${EMBEDDING_ID}.zip \
     --load best_model.save \
     --dir experiments/$DATASET/${MODEL_DIR}/"

echo "Running test prediction: $test_cmd"
eval $test_cmd

# Ensure output directories for evaluation
mkdir -p experiments/$DATASET/${MODEL_DIR}/evaluation/res/$DATASET
mkdir -p experiments/$DATASET/${MODEL_DIR}/evaluation/ref/data/$DATASET

# Copy predictions.json to the expected location for evaluation
cp experiments/$DATASET/${MODEL_DIR}/test.conllu.json experiments/$DATASET/${MODEL_DIR}/evaluation/res/$DATASET/predictions.json

# Copy gold test data to the expected location for evaluation
cp ../../data/$DATASET/test.json experiments/$DATASET/${MODEL_DIR}/evaluation/ref/data/$DATASET/test.json

# Run evaluation script
echo "Running evaluation on test predictions..."
python ../../evaluation/evaluate_single_dataset.py \
    experiments/$DATASET/${MODEL_DIR}/evaluation/ref/data/$DATASET/test.json \
    experiments/$DATASET/${MODEL_DIR}/evaluation/res/$DATASET/predictions.json

echo "Test evaluation completed."
echo "--------------------------------------"
echo "Final F1 scores:"
echo "Dev: $(grep -o 'Primary Dev F1 on epoch.*is.*' experiments/$DATASET/${MODEL_DIR}/logs.txt | tail -1 | sed 's/.*is //')"
echo "Test: See above for Sentiment Tuple F1 score"
echo "--------------------------------------"