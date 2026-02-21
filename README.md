# Support Ticket Classification System
# Project Overview:

This project implements a Machine Learning-based system to automatically classify customer support tickets into predefined categories using Natural Language Processing (NLP).

The system analyzes ticket text and predicts the appropriate category, helping automate ticket routing and improve support operations efficiency.

# Problem Statement:

Support teams receive a large volume of customer tickets daily. Manually reading and categorizing these tickets:

Consumes time

Delays issue resolution

Reduces operational efficiency

Makes scaling difficult

This project solves the problem by building an automated ticket classification system using supervised machine learning.

# Dataset Description:

dataset link: https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset 

The dataset contains two columns:

Column Name	Description
Document	Support ticket text
Topic_group	Ticket category label

Each ticket belongs to one predefined category.

# Project Workflow:

 1️⃣ Data Preprocessing:

Removed missing values

Converted text to lowercase

Removed punctuation and special characters

Removed stopwords

Cleaned noisy text

  2️⃣ Feature Engineering:

Applied TF-IDF Vectorization

Used unigrams and bigrams

Converted text into numerical format for model training

   3️⃣ Model Training:

Split dataset into training and testing sets

Used LinearSVC (Support Vector Machine) for multi-class classification

Trained model on processed text data

   4️⃣ Model Evaluation:

The model was evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

The model demonstrated strong performance in categorizing unseen support tickets.

# How the System Works:

User Input (Ticket Text)

→ Text Cleaning

→ TF-IDF Transformation

→ LinearSVC Model

→ Predicted Category

# Streamlit Web Application:

A simple and interactive web interface was built using Streamlit.

Features:

User enters ticket description

Model predicts category instantly

Real-time classification output

# Tech Stack:

Python

Pandas

NumPy

Scikit-learn

NLTK

TF-IDF

LinearSVC

Streamlit


# Project Structure:

support-ticket-classification/

├── all_tickets_processed_improved_v3.csv

├── ML_Task_2 (1).ipynb

├──ticket_model.pkl

├── app.py

├── requirements.txt

└── README.md

# How to Run the Project:

 Run Streamlit App:

streamlit run app.py

# Business Impact:

Automates ticket categorization

Reduces manual workload

Improves routing efficiency

Enables scalable support operations

Enhances response time

# Future Improvements:

Add priority prediction model

Deploy on cloud (Streamlit Cloud / AWS)

Integrate with helpdesk systems

Use advanced NLP models (BERT, Transformers)

Add confidence score visualization

# Conclusion:

This project demonstrates how NLP and Machine Learning can be used to automate real-world support workflows. By converting raw ticket text into structured categories, the system improves operational efficiency and scalability.