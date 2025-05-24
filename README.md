# PySpark Amazon Book Reviews Analysis

This project applies distributed machine learning techniques using **Apache Spark MLlib** on the **Amazon Books Review dataset**, which includes 3 million user reviews of over 200,000 unique books. The goal is to build classification and recommendation models that predict review ratings and recommend books efficiently in a large-scale data environment.

## 📌 Project Scope

This repository contains the **PySpark-only implementation**, focusing on:
- **Binary and Multiclass Classification** for rating prediction
- **ALS-based Recommendation System**
- **Distributed preprocessing and feature engineering** using PySpark
- **Performance evaluation** in a local standalone Spark setup

## Dataset

Dataset used: [Amazon Books Reviews - Kaggle](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews)  
- `Books_rating.csv`: ~3M reviews
- `Books_data.csv`: metadata for 212,404 books

## Environment

- **Apache Spark 3.5.4**
- Local machine: 32GB RAM, Intel i7 @ 2.30GHz (8 cores)
- Python 3.10
- PySpark

## Models Used

### Classification
- **Multiclass**: Logistic Regression, Random Forest, Decision Tree
- **Binary**: Linear SVC, Logistic Regression, Random Forest, Decision Tree

### Recommendation
- **ALS (Alternating Least Squares)** collaborative filtering

## Model Results

### Multiclass Classification

| Model              | Accuracy | Precision | Recall | F1 Score | Training Time (s) |
|--------------------|----------|-----------|--------|----------|-------------------|
| Logistic Regression| 0.65     | 0.59      | 0.65   | 0.59     | 176.08            |
| Decision Tree      | 0.60     | 0.46      | 0.60   | 0.48     | 168.71            |
| Random Forest      | 0.59     | 0.35      | 0.59   | 0.44     | 180.17            |

---

### Binary Classification  
(Review scores: 1-3 ➝ 0, 4-5 ➝ 1)

| Model              | Accuracy | Precision | Recall | F1 Score | Training Time (s) |
|--------------------|----------|-----------|--------|----------|-------------------|
| Linear SVC         | 0.85     | 0.84      | 0.85   | 0.84     | 247.73            |
| Logistic Regression| 0.85     | 0.84      | 0.85   | 0.84     | 215.41            |
| Decision Tree      | 0.79     | 0.77      | 0.79   | 0.77     | 244.25            |
| Random Forest      | 0.77     | 0.59      | 0.77   | 0.67     | 180.52            |

> ✅ Binary classification models consistently outperformed multiclass models in both accuracy and F1 scores.

## 📊 Highlights

- Full utilization of 8-core CPU and 32GB RAM on local Spark
- Efficient sparse feature handling with TF-IDF + MinMaxScaler
- ALS-based collaborative filtering shows promising RMSE on test set
- PySpark MLlib scales better than Google Colab in this setup

