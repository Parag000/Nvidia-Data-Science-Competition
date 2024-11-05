# 👻 Predicting the Scariest Monster - Nvidia Hackathon

## Link to Colab Notebook:
[Open this notebook in Google Colab](https://colab.research.google.com/drive/1N9I19hw8kbnNC2j3QmP5nTD5GbgGF5qW?usp=sharing)

## Our Submission Scores
![Submission Score](https://github.com/Parag000/Nvidia-Data-Science-Competition/blob/main/submisssion-score.png?raw=true)

## Project Overview

This project presents a solution for the **ODSC 2024 NVIDIA Hackathon**, where data scientists are challenged to predict the "Scariest Monster" using a massive dataset filled with 12 million entries, each described by 106 anonymous features. The ultimate goal is to forecast the number of votes each monster received in a global terror poll, utilizing GPU-accelerated data processing and machine learning techniques.

## Dataset

The competition dataset includes:

- **12 million monster entries**
- **106 anonymous features** (a mix of categorical and numerical)
- **Target variable 'y'**: Number of votes each monster received in the global terror poll
- **Dataset size**: Approximately **8-10GB**

## Approach

Our approach to tackling this challenge involves the following steps:

1. **Data Loading and Preprocessing**: 
   - Loading the data using **cuDF** (RAPIDS NVIDIA API) for GPU-accelerated processing.
   - Performing basic **Exploratory Data Analysis (EDA)** to understand the dataset.
   - Dropping categorical columns to avoid creating sparse matrices.
   - Applying **mean imputation** for numerical columns.
   - Removing outliers and performing **robust normalization** for stability.

2. **Memory-Efficient Train-Test Split**:
   - Creating a custom train-test split method to handle memory constraints effectively.
   - Using a random shuffled column for efficient data shuffling and splitting.

3. **Model Training**:
   - Implementing a **Random Forest Regressor** using the RAPIDS **cuML** library for GPU-accelerated processing.

4. **Post-processing**:
   - Applying **inverse robust scaling** to calculate the final RMSE value.

5. **Prediction and Submission**:
   - Generating predictions on the test set.
   - Preparing the submission file in accordance with the competition guidelines.

## Technologies Used

- **Used A100 hardware acceleration**
- **Python 3.x**
- **RAPIDS cuDF** for GPU-accelerated data processing
- **RAPIDS cuML** for GPU-accelerated machine learning
- **Scikit-learn** for preprocessing and metrics
- **Google Colab Notebook** for interactive development

## Results

The model's performance is evaluated based on **Root Mean Squared Error (RMSE)**, with lower scores indicating better performance.


## Getting Started

1. **Clone this repository**:
   ```bash
   https://github.com/Parag000/Nvidia-Data-Science-Competition.git


## Leaderboard
![Leaderboard](https://github.com/Parag000/Nvidia-Data-Science-Competition/blob/main/leaderboard.png)

