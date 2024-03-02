# Disaster Image Classification Task

This repository contains code for a disaster image classification task based on the paper [Paper Link](https://arxiv.org/pdf/2107.01284v1.pdf). The task involves classifying images into six main categories of disasters using a pre-trained ResNet-50 architecture. The code consists of several Python scripts for data preprocessing, model training, evaluation, and comparison with the results reported in the paper.

## Dataset
The dataset used for this task is the Comprehensive Disaster Dataset (CDD). It consists of images categorized into six main disaster categories: Fire_Disaster, Human_Damage, Water_Disaster, Land_Disaster, Damaged_Infrastructure, and Non_Damage. The dataset can be downloaded from [Dataset Link](https://drive.google.com/drive/folders/1VvkBRIYW6oD31K3gkPk4-4nlGE2poXFU).

## Project Structure
- `cli.py`: Main script to execute the project. It orchestrates data processing, model training, evaluation, and cross-validation.
- `configs/config.yaml`: Configuration file containing settings for the project.
- `data/data_processing.py`: Module for data loading, preprocessing, and dataset creation.
- `models/model_training.py`: Module for model training, testing, and evaluation.
- `utils/config.py`: Configuration class to load settings from the YAML file.
- `utils/utility.py`: Utility functions for displaying comparison tables, merging datasets, and performing cross-validation.

## Usage
1. Clone the repository to your local machine.
2. Download the Comprehensive Disaster Dataset (CDD) and extract it into the appropriate directory.
4. Load and edit the (`configs/config.yaml`) file for customization options such as batch size, learning rate, and number of epochs.
3. Run the `cli.py` script to train the model, evaluate it, and compare the results with the paper.

## Results
The trained model's performance is evaluated on the test set, and the results are compared with those reported in the paper. Additionally, cross-validation is performed to assess the model's robustness.

## References
- Link to Paper: [Paper Link](https://arxiv.org/pdf/2107.01284v1.pdf)
- Link to Dataset: [Dataset Link](https://drive.google.com/drive/folders/1VvkBRIYW6oD31K3gkPk4-4nlGE2poXFU)

