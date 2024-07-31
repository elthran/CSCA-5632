# CSCA-5632-Final: Book Recommendation System Using Unsupervised Learning

## Project Overview

This project aims to build a book recommendation system using unsupervised machine learning techniques. The dataset for this project is sourced from [Kaggle's Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data). The goal is to recommend books to users based on the data available.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
    - [Step 1: Gather Data](#step-1-gather-data)
    - [Step 2: Identify an Unsupervised Learning Problem](#step-2-identify-an-unsupervised-learning-problem)
    - [Step 3: Exploratory Data Analysis (EDA)](#step-3-exploratory-data-analysis-eda)
    - [Step 4: Perform Analysis Using Unsupervised Learning Models](#step-4-perform-analysis-using-unsupervised-learning-models)
4. [Conclusions](#conclusions)
5. [Summary](#summary)
6. [Challenges and Future Work](#challenges-and-future-work)
7. [References](#references)

## Introduction

In this project, we will develop a book recommendation system using unsupervised learning techniques. The primary aim is to explore different models and approaches to provide personalized book recommendations to users based on the available dataset. The focus will be on model building, training, and evaluation, ensuring that the methods are tailored to handle the dataset's characteristics.

## Dataset

The dataset used for this project is obtained from Kaggle and includes information about books, users, and their interactions. The dataset can be found [here](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data).

## Methodology

### Step 1: Gather Data

- **Objective:** Select a data source and problem.
- **Data Source:** [Kaggle Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data).
- **Provenance:** The dataset is collected and shared by Arash Nick on Kaggle. It includes user ratings, book details, and user demographic information.
- **Discussion:** The dataset is suitable for building a recommendation system as it provides comprehensive information about user interactions with books.

### Step 2: Identify an Unsupervised Learning Problem

- **Objective:** Focus on model building and training using unsupervised learning techniques.
- **Approach:** Use clustering algorithms to identify patterns and similarities among books and users.
- **Models to Explore:** K-Means Clustering, Hierarchical Clustering, DBSCAN, etc.
- **Additional Research:** Explore research papers and existing algorithms to enhance model performance and comparison.

### Step 3: Exploratory Data Analysis (EDA)

- **Objective:** Inspect, visualize, and clean the data to prepare it for modeling.
- **Procedures:**
  - **Data Description:** Describe factors (features) such as book titles, authors, user ratings, etc.
  - **Visualizations:** Use box-plots, scatter plots, histograms to describe data distribution.
  - **Correlation Analysis:** Identify correlations between different factors.
  - **Data Transformation:** Normalize or scale data if necessary.
  - **Outlier Detection:** Identify and handle outliers or missing values.
  - **Feature Importance:** Hypothesize which factors might be more important for the recommendation system.

### Step 4: Perform Analysis Using Unsupervised Learning Models

- **Objective:** Build and train models, compare their performance, and present findings.
- **Models to Use:**
  - **Singular Value Decomposition (SVD):** Reduce the dimensionality of the interaction matrix and identify latent factors.
  - **Principal Component Analysis (PCA):** Reduce dimensionality and visualize clusters.
- **Hyperparameter Optimization:** Tune model parameters to achieve optimal performance.
- **Comparison and Evaluation:** Compare models and justify their performance and limitations.

## Conclusions

- **Random Model:** The Random Model serves as a baseline with an RMSE of 2.30, MAE of 4.44, precision of 36.75%, recall of 59.98%, and an F1-score of 45.58%. This model performs poorly as expected, given its random nature.
- **Global Average Model:** This model shows a slight improvement in RMSE (1.99) and MAE (3.80) compared to the Random Model. However, it achieves perfect precision (100%) but zero recall and F1-score, indicating that while the model's predictions were correct, it failed to recall any relevant items.
- **User Average Model:** The User Average Model further improves the RMSE to 1.92 and MAE to 2.74. It also shows better balance in precision (56.69%), recall (46.88%), and F1-score (51.32%) compared to the previous models.
- **Item Average Model:** Similar to the Global Average Model, this model has an RMSE of 2.02 and an MAE of 3.91. It achieves perfect precision (100%) but zero recall and F1-score, highlighting the same limitations as the Global Average Model.
- **Collaborative Filtering Model:** This model has the highest RMSE (5.51) and MAE (4.69) among all models, indicating poor performance. Its precision is 42.11%, but it struggles with recall (3.81%) and F1-score (6.99%), showing that it fails to accurately recommend relevant items.
- **SVD-based Collaborative Filtering Model:** The SVD-based Collaborative Filtering Model demonstrates the best overall performance with an RMSE of 1.96 and MAE of 2.97. It achieves the highest precision (60.87%), recall (50.87%), and F1-score (55.42%) among all models, indicating a balanced and effective recommendation system.

## Summary

The SVD-based Collaborative Filtering Model is the best-performing model, demonstrating the lowest error rates and the highest precision, recall, and F1-score. This indicates that SVD is effective in capturing latent factors in the dataset, leading to more accurate and relevant book recommendations.

## Challenges and Future Work

- **Data Sparsity:** The dataset's sparsity posed a challenge, affecting the performance of some models, particularly the Collaborative Filtering Model.
- **Model Optimization:** Further tuning of hyperparameters and exploring advanced techniques could potentially enhance model performance.
- **Additional Models:** Investigating other models such as deep learning-based approaches or hybrid models combining collaborative and content-based filtering could provide further improvements.

## References

- Kaggle Dataset: [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data)
- Research Papers and Articles on Recommendation Systems and Unsupervised Learning.

---

This `README.md` provides a structured plan and overview for the project, guiding through each step of the process from data gathering to model evaluation. Follow this methodology to ensure a comprehensive approach to building your book recommendation system.
