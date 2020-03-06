# Overview

A bi-directional GRU implementation for Relation Extraction:

The GRU is loosely based on the approach done in the work of Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification (Zhou et. al, 2016):

1. Bidirectional GRU
2. Attention layer
3. L2 Regularization

# Installation
This project is implemented in python 3.6 and tensorflow 2.0. Follow these steps to setup your environment:

1. [Download and install Conda](http://https://conda.io/projects/conda/en/latest/user-guide/install/index.html "Download and install Conda")
2. Create a Conda environment with Python 3.6

```
conda create -n nlp-hw4 python=3.6
```

3. Activate the Conda environment.
```
conda activate nlp-hw4
```
4. Install the requirements:
```
pip install -r requirements.txt
```

5. Download spacy model
```
python -m spacy download en_core_web_sm
```

6. Download glove wordvectors:
```
./download_glove.sh
```

# Data

Training can be found in `data/train.txt` and validation is in `data/val.txt`. We are using data from a previous SemEval shared task which in total had 8,000 training examples. The train/validation examples are a 90/10 split from this original 8,000. More details of the data can be found in the overview paper SemEval-2010 task 8: multi-way classification of semantic relations between pairs of nominals (Hendrickx et. al, 2009) as well as extra PDFs explaining the details of each relation in the dataset directory.

## Train and Predict

There are 4 main scripts in the repository `train_basic.py`, `train_advanced.py`, `predict.py` and `evaluate.pl`.

- Train scripts do as described and saves the model to be used later for prediction. Basic training script trains the basic `MyBasicAttentiveBiGRU` model implementation (Bi-RNN+attention).

- Predict generates predictions on the test set `test.txt` and saves your output to a file. You will submit your predictions to be scored against the hidden (labels) test set. Both files are set with reasonable defaults in their arguments but example commands are shown below.

- Evaluation script is the pearl script unlike others. You can use it see detailed report of your predictions file against the gold labels.


#### Train a model
```
python train.py --embed-file embeddings/glove.6B.100D.txt --embed-dim 100 --batch-size 10 --num_epochs 5

# stores the model by default at : serialization_dirs/basic/
```

#### Predict with model
```
python predict.py --prediction-file my_predictions.txt --batch-size 10
```
