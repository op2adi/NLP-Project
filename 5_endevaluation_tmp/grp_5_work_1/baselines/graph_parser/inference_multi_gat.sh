#!/bin/bash
# This script runs inference using the enhanced multi-attention GAT model for sentiment graph parsing

# Parameters
DATASET=${1:-"norec"}  # Default to norec if not specified
EMBEDDING_ID=${2:-"32"}  # Default to 32 if not specified
GAT_HEADS=${3:-8}       # Default to 8 attention heads
GAT_DROPOUT=${4:-0.15}  # Default to 0.15 dropout rate

# Define directories
MODEL_DIR="experiments/$DATASET/gat"
TEST_FILE="sentiment_graphs/$DATASET/test.conllu"
EMBEDDINGS="embeddings/${EMBEDDING_ID}.zip"
MODEL_PATH="$MODEL_DIR/best_model.save"
OUTPUT_DIR="$MODEL_DIR"  # Use the same directory as the model to avoid vocabs.pk issues

echo "Running inference on $DATASET test data using GAT model"
echo "--------------------------------------"
echo "Model directory: $MODEL_DIR"
echo "Model path: $MODEL_PATH"
echo "Test file: $TEST_FILE"
echo "Embeddings: $EMBEDDINGS"
echo "GAT Heads: $GAT_HEADS"
echo "GAT Dropout: $GAT_DROPOUT"
echo "Output directory: $OUTPUT_DIR"
echo "--------------------------------------"

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file $TEST_FILE not found!"
    echo "Make sure you've prepared the data first by running the conversion script."
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file $MODEL_PATH not found!"
    echo "You need to train the model first using train_gat.sh script."
    exit 1
fi

# Check if vocabs.pk exists
if [ ! -f "$MODEL_DIR/vocabs.pk" ]; then
    echo "Error: Vocabulary file $MODEL_DIR/vocabs.pk not found!"
    echo "The model directory is missing necessary files."
    exit 1
fi

# Create evaluation directories
mkdir -p $OUTPUT_DIR/evaluation/res/$DATASET
mkdir -p $OUTPUT_DIR/evaluation/ref/data/$DATASET

# Run inference using the best model
echo "Running inference..."
python src/main.py --config configs/sgraph.cfg --bridge gat \
     --gat_heads $GAT_HEADS --gat_dropout $GAT_DROPOUT \
     --predict_file $TEST_FILE \
     --external $EMBEDDINGS \
     --load $MODEL_PATH \
     --dir $OUTPUT_DIR \
     --other_target_style none \
     --target_style scope

# Check if the prediction file was generated
if [ ! -f "$OUTPUT_DIR/test.conllu.json" ]; then
    echo "Error: Prediction file $OUTPUT_DIR/test.conllu.json was not generated!"
    echo "Inference failed. Check the logs above for details."
    exit 1
fi

# Copy predictions and gold test data for evaluation
echo "Preparing for evaluation..."
cp $OUTPUT_DIR/test.conllu.json $OUTPUT_DIR/evaluation/res/$DATASET/predictions.json
cp ../../data/$DATASET/test.json $OUTPUT_DIR/evaluation/ref/data/$DATASET/test.json

# Run evaluation script
echo "Running evaluation on test predictions..."
python ../../evaluation/evaluate_single_dataset.py \
    $OUTPUT_DIR/evaluation/ref/data/$DATASET/test.json \
    $OUTPUT_DIR/evaluation/res/$DATASET/predictions.json

echo "Inference completed."
echo "--------------------------------------"
echo "Results are available at: $OUTPUT_DIR/test.conllu.json"
echo "F1 scores are shown above in the Sentiment Tuple F1 scores"
echo "--------------------------------------"