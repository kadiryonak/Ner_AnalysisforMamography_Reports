# Ner_AnalysisforMamography_Reports
Welcome to the NER (Named Entity Recognition) Model with Curriculum Learning project! This repository contains a deep learning model designed to detect and classify named entities in text data using Curriculum Learning. The project is built using PyTorch and utilizes a custom dataset for training the model. The core goal is to improve the training process by gradually introducing more difficult samples, which helps the model learn more effectively.

##Project Overview
This project implements a NER model that leverages Curriculum Learning to train on easier samples first and progressively increase the difficulty level of the data. This approach allows for more efficient model training and potentially better generalization. The model is trained using mixed-precision training and evaluated based on F1 score.

##The repository contains the following modules:
curriculum_learning.py: Implements the logic for splitting data into difficulty levels and performing training in stages.
dataset.py: Defines the custom dataset class for NER data processing.
data_processing.py: Handles data preparation and text-token mappings.
model.py: Defines the architecture of the NER model.
training.py: Includes the training loop with mixed-precision training and evaluation metrics.
main.py: The main entry point to run the model training and evaluation.
##Dataset Information
The dataset used for this project consists of BIO-tagged sequences. The data is split into three stages based on the difficulty level, which is determined by the length of the sequences. The custom NERDataset class helps load the data efficiently for training.

##Model Evaluation
The model evaluation is based on the following metrics:

Accuracy
Precision
Recall
F1 Score
Example performance of the model:

| Model              | Accuracy | F1 Score |
|--------------------|----------|----------|
| NER AttenitonLSTM  | 0.98     | 0.97     |

##Features
Curriculum Learning for staged training based on data difficulty
Mixed precision training for optimized performance
Custom NER dataset class with efficient data loading
Evaluation of the model based on standard classification metrics
Plotting of F1 scores across epochs for visual performance tracking

##Prerequisites
Before you can run the project, you’ll need to install the following software and dependencies:

Python 3.x
PyTorch
torchvision
PIL (Python Imaging Library)
pandas
scikit-learn

##Installation
Clone this repository to your local machine:
git clone https://github.com/your-repository/Ner_AnalysisforMamography_Reports
cd Ner_AnalysisforMamography_Reports
pip install -r requirements.txt

##Running the Project
Once you have installed the necessary dependencies, you can run the model training with the following command:
python main.py

##Curriculum Learning Workflow
Data Splitting: The data is split into three levels of difficulty (easy, medium, hard) based on the length of the BIO sequences.
Staged Training: The model is first trained on the easy data, then on medium difficulty, and finally on the hard data.
Mixed Precision Training: The model utilizes mixed-precision training to optimize memory usage and speed.
Evaluation: At the end of each training stage, the model's F1 score is calculated and plotted across epochs.

##License
This project is licensed under the MIT License - see the LICENSE file for details.
