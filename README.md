# Deep Learning Project 6 – Image Retrieval

## Overview

TBA

## Dataset

We use the CIFAR-10 dataset, which contains 60,000 color images from 10 classes:

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

## Retrieval Protocol

* Gallery set: CIFAR-10 training set (50,000 images)
* Query set: CIFAR-10 test set (10,000 images)
* Relevance criterion: Images belonging to the same class are considered relevant.

## Baseline Method

The baseline retrieval system uses:

* Pretrained ResNet-18 (ImageNet weights)
* Feature extraction from the backbone network
* L2 normalization of feature vectors
* Cosine similarity for image retrieval

For each query image, cosine similarity is computed between the query embedding and all gallery embeddings. The gallery images are then ranked according to similarity.

## Project Structure

```text
src/
├── dataset.py
├── extract_features.py
└── retrieval.py
```

