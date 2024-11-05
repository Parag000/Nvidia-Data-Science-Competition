# Nvidia-Data-Science-Competition
Submission for NVIDIA's Open Data Science Competition: A regression model predicting the scariest monster based on a dataset of 12 million records with 106 anonymous features. Utilizes GPU-accelerated data frame processing and model training via NVIDIA RAPIDS API to forecast global terror poll votes, evaluated using RMSE.

# 🎃 NVIDIA Open Data Science Competition: Predicting the Scariest Monster 🧟‍♀️👻

This repository contains my submission for NVIDIA's Open Data Science Competition, where the goal is to identify the scariest monster based on a dataset of 12 million records with 106 anonymous features. Using GPU-accelerated data processing and model training through the NVIDIA RAPIDS API, this project aims to predict global terror poll votes for each monster with high accuracy.

## 📊 Project Overview

### Problem Statement
The dataset includes information on 12 million unique monsters worldwide, each described by a combination of 106 categorical and numerical features. The challenge is to predict the target variable, **"y"** — the number of votes each monster received in the global terror poll.

The **objective** is to create a regression model that accurately forecasts the "y" value, helping us answer: "Who is the scariest monster of them all?" 🏆💀

### Key Components
- **Data**: 12 million records with anonymous features describing various attributes of each monster.
- **Goal**: Predict the number of votes each monster received.
- **Evaluation Metric**: **Root Mean Squared Error (RMSE)**, with lower scores indicating better model performance. Rankings are based on a private leaderboard evaluated on a hidden subset of the test data.

---

## 🚀 Solution Approach

1. **Data Preprocessing**
   - Loaded and preprocessed the dataset using **cuDF** for efficient GPU-accelerated data manipulation.
   - Split the dataset into training and testing sets using **cuml.model_selection.train_test_split**.

2. **Feature Engineering**
   - Applied necessary transformations and feature engineering techniques to enhance the model's predictive capability.

3. **Modeling**
   - Built a regression model using **cuml.ensemble.RandomForestRegressor**, leveraging the GPU for accelerated training.
   - Evaluated model performance using the **mean_squared_error** metric.

4. **Evaluation**
   - Calculated RMSE to assess the model's performance on the test set.

---

## 🛠️ Tech Stack

- **NVIDIA RAPIDS**: GPU-accelerated data processing with cuDF for data manipulation, enabling faster computations compared to traditional CPU-based pandas.
- **cuML**: Used for machine learning tasks, including regression modeling and evaluation metrics.
- **NumPy**: Utilized for numerical operations.
- **CuPy**: Used for GPU-accelerated array operations.

---

## 🗂️ Repository Structure

- **Notebook**: The main Jupyter Notebook containing all steps, from data loading to model evaluation.
- **Scripts**: Modularized scripts for data handling, feature engineering, and model training.
- **Results**: Output files or summary metrics generated from model evaluation.

---

## 📈 Results & Next Steps
- **Final RMSE Score**: _<639.85>_
- **Future Improvements**:
  - Explore different feature engineering techniques.
  - Fine-tune hyperparameters further.
  - Experiment with ensemble models for potential performance gains.

---

## ⚡ Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/nvidia-monster-prediction.git

