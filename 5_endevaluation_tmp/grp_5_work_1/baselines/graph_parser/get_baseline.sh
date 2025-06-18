#!/bin/bash
# Set some random seeds that will be the same for all experiments
SEEDS=(17181920)

# Setup directories
mkdir logs
mkdir experiments

python temp.py

# # Convert json files to conllu for training
# # Currently only creates head_final, but you can
# # experiment with other graph setups by expanding this section
# for DATASET in darmstadt_unis mpqa multibooked_ca multibooked_eu norec opener_es opener_en; do
#     python3 convert_to_conllu.py --json_dir ../../data/"$DATASET" --out_dir sentiment_graphs/"$DATASET" --setup head_final
# done;

# Download word vectors
if [ -d embeddings ]; then
    echo "Using downloaded word embeddings"
else
    mkdir embeddings
    cd embeddings
    wget http://vectors.nlpl.eu/repository/20/58.zip
    wget http://vectors.nlpl.eu/repository/20/32.zip
    wget http://vectors.nlpl.eu/repository/20/34.zip
    wget http://vectors.nlpl.eu/repository/20/18.zip
    wget http://vectors.nlpl.eu/repository/20/68.zip
cd ..
fi

# # Check if PyTorch Geometric is installed (required for GAT)
# pip list | grep -q torch-geometric
# if [ $? -ne 0 ]; then
#     echo "Installing PyTorch Geometric for GAT support..."
#     pip install torch-geometric
# fi

# # Iterate over datsets
# for DATASET in multibooked_ca; do
#     mkdir -p logs/$DATASET;
#     mkdir -p experiments/$DATASET;
    
#     # Create directories for different bridge methods
#     for BRIDGE in head_final gat_bridge; do
#         mkdir -p logs/$DATASET/$BRIDGE
#         mkdir -p experiments/$DATASET/$BRIDGE
#     done
    
#     # Run the standard model
#     echo "Running $DATASET - head_final"
#     mkdir -p experiments/$DATASET/head_final
#     mkdir -p logs/$DATASET/head_final
#     python src/main.py --config configs/sgraph.cfg --train sentiment_graphs/$DATASET/train.conllu --val sentiment_graphs/$DATASET/dev.conllu --external embeddings/32.zip --dir experiments/$DATASET/head_final/
    
#     # Run with GAT bridge model
#     echo "Running $DATASET - gat_bridge"
#     mkdir -p experiments/$DATASET/gat_bridge
#     mkdir -p logs/$DATASET/gat_bridge
#     python src/main.py --config configs/sgraph.cfg --bridge gat --gat_heads 4 --gat_dropout 0.1 --train sentiment_graphs/$DATASET/train.conllu --val sentiment_graphs/$DATASET/dev.conllu --external embeddings/32.zip --dir experiments/$DATASET/gat_bridge/
# done


for DATASET in cross_lang; do
    python3 convert_to_conllu.py --json_dir ../../data/"$DATASET" --out_dir sentiment_graphs/"$DATASET" --setup head_final
done;