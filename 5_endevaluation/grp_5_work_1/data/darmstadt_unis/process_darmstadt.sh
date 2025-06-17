#!/bin/bash

###############################################################################
# COLLECT DATA
###############################################################################

# Download the Darmstadt data from the following URL:
# https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2448

# Use curl to download the zip file (the actual file link must be resolved)
echo "Downloading DarmstadtServiceReviewCorpus.zip..."

curl -L -o DarmstadtServiceReviewCorpus.zip "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2448/DarmstadtServiceReviewCorpus.zip?sequence=5&isAllowed=y"

###############################################################################
# PROCESS DATA
###############################################################################

# Process darmstadt data
unzip DarmstadtServiceReviewCorpus.zip
cd DarmstadtServiceReviewCorpus
unzip universities
grep -rl "&" universities/basedata | xargs sed -i 's/&/and/g'
cd ..
python3 process_darmstadt.py
rm -rf DarmstadtServiceReviewCorpus
