# Topological image feature extraction for Visual Transformer Breast Cancer Classification

## Introduction

This notebook implements the experiments proposed on [my Msc. thesis](https://github.com/EstherRH00/TDA_Visual_Transformer/blob/main/TFM.pdf) and aims to answer the following hypothesis:

**H1:** Image preprocessing techniques (noise reduction and contrast enhancement) positively impact the classification accuracy of a Vision Transformer model.

**H2:** The incorporation of topological descriptors derived from TDA provides complementary structural information that improves classification performance compared to a standard Vision Transformer.

To do so, it uses the mammographies from the [CBIS-DDSM dataset](www.cancerimagingarchive.net/collection/cbis-ddsm/), the [base-sized Visual transformer](https://huggingface.co/google/vit-base-patch16-224), the [Gudhi library](https://gudhi.inria.fr/python/latest/) and [giotto-tda](https://giotto-ai.github.io/gtda-docs/latest/) to extract topological descriptors, as well as pytorch and other python libraries.

Roughly, the project structure is:


```
.
├── data                    <- Images and csv files
├── src                     <- Model and processing implementations
├── checkpoints             <- Experiment results
├── .gitignore
├── notebook.ipynb          <- Explains the project and triggers all experiments
├── notebook.html           <- Notebook, exported to html 
├── README.md               <- Setup instructions and brief description of the project
├── requirements.txt        <- List of dependencies
└── TFM.pdf                 <- Msc. Thesis
```

## Results

Seven experiments were run across two hypotheses. **H1 is supported**: image preprocessing (CLAHE + Gaussian denoise) dramatically improves ViT classification. Aggressive data augmentation without data expansion hurts rather than helps. **H2 is not supported**: none of the four TDA fusion experiments (persistence images via dual-ViT, or compact vector descriptors via linear fusion) improved over the preprocessed ViT baseline. 

## Setup

1. Install all dependencies: 
    * `pip install -r requirements.txt`

2. Download data and store it in `/data` folder: to download the dataset, the [CBIS-DDSM dataset](www.cancerimagingarchive.net/collection/cbis-ddsm/) webpage provides four csv files and a `.tcia` file. That file is used to download the images using the [TCIA image retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images), that is a program that you will need to install. Image retrieval for this dataset took about 8hrs, so it is recommended to do so overnight. 

3. Update the `BASE_PATH` variable in `src/utils/image_utils.py` to point to the folder where the CBIS-DDSM DICOM images are stored on your system.

## Licensing and citations

This project uses the CBIS-DDSM dataset, licensed under CC BY 3.0.
    Citation: Sawyer-Lee et al. (2016), DOI: 10.7937/K9/TCIA.2016.7O02S9CY
