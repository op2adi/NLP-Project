#!/bin/bash


###############################################################################
# COLLECT DATA
###############################################################################

# First download the mpqa 2.0 data from http://mpqa.cs.pitt.edu/corpora/mpqa_corpus/mpqa_corpus_2_0 and change the following path to point to the tar file

if [ ! -f "$mpqa_tar_file" ]; then
    echo "Downloading MPQA 2.0 dataset..."
    curl -L -o $mpqa_tar_file "https://mpqa.cs.pitt.edu/corpora/mpqa_corpus/mpqa_corpus_2_0/mpqa_2_0_database.tar.gz"
else
    echo "$mpqa_tar_file already exists. Skipping download."
fi
mpqa_tar_file="./mpqa_2_0_database.tar.gz"
tar -xvf $mpqa_tar_file


###############################################################################
# PROCESS DATA
###############################################################################

# Process mpqa data
python3 process_mpqa.py
