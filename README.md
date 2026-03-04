#  Support Ticket Classification System

## Project Overview

This project implements a Machine Learning-based system to automatically classify customer support tickets into predefined categories using Natural Language Processing (NLP).

The system analyses ticket text and predicts the appropriate category, helping automate ticket routing and improving support operations efficiency.

---

## Problem Statement

Support teams receive a large volume of customer tickets daily. Manually reading and categorising these tickets:

-  Consumes time
-  Delays issue resolution
-  Reduces operational efficiency
-  Makes scaling difficult

This project solves the problem by building an automated ticket classification system using supervised machine learning.

---

## Dataset

**Source:** [Kaggle – Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)

| Column | Description |
|---|---|
| `Document` | Support ticket text |
| `Topic_group` | Ticket category label |

The dataset contains **3 806 labelled tickets** across **8 categories**.

---

## Project Workflow

### 1️⃣ Data Preprocessing
- Removed missing values
- Converted text to lowercase
- Removed punctuation, digits, and special characters
- Removed NLTK English stopwords
- Filtered out short noise words (< 3 characters)

### 2️⃣ Feature Engineering
- Applied **TF-IDF Vectorisation** (unigrams + bigrams, top 20 000 features)
- Converted text into a numerical matrix for model training

### 3️⃣ Model Training
- Split dataset into training and testing sets
- Used **LinearSVC** wrapped in an sklearn **Pipeline** for end-to-end classification

### 4️⃣ Model Evaluation
The model was evaluated using:
- Accuracy
- Precision / Recall / F1-score
- Confusion Matrix

---

## Supported Categories

| Icon | Category |
|---|---|
| 🖥️ | Hardware |
| 🔑 | Access |
| 📦 | Miscellaneous |
| 👥 | HR Support |
| 🛒 | Purchase |
| 🛡️ | Administrative rights |
| 💾 | Storage |
| 📋 | Internal Project |

---

## Streamlit Web Application

A beautiful, interactive web interface built with Streamlit featuring:

-  **Live ticket classification** with instant predictions
-  **Example prompts** to try the model quickly
-  **Dataset overview** with an Altair bar chart
-  **Sample ticket browser**
- ℹ **About tab** explaining the model pipeline

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.x | Core language |
| scikit-learn | TF-IDF + LinearSVC pipeline |
| NLTK | Stopword removal |
| Streamlit | Web application frontend |
| Altair | Interactive charts |
| Pandas / NumPy | Data handling |
| joblib | Model serialisation |

---

## Project Structure

```
ML_TASK_2/
├── all_tickets_processed_improved_v3.csv   # Processed dataset
├── ML_Task_2.ipynb                         # Training notebook
├── ticket_model.pkl                        # Trained sklearn Pipeline
├── app.py                                  # Streamlit application
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

---

## User Interface

![img alt](https://github.com/SathishB-1/Support_Ticket_Classification_System/blob/efb472e00ecaff5a7c763d454c67c0ea4949f368/Screenshot%202026-03-04%20202632.png)

---

## Business Impact

| Benefit | Description |
|---|---|
|  Speed | Instant ticket categorisation |
|  Accuracy | ML-driven, consistent predictions |
|  Scalability | Handles high ticket volumes |
|  Cost saving | Reduces manual triage effort |
|  Better routing | Tickets reach the right team faster |

---

## Future Improvements

-  Add priority prediction model
-  Show confidence scores for each category
-  Deploy on Streamlit Cloud / AWS
-  Integrate with helpdesk APIs (Zendesk, Freshdesk)
-  Replace LinearSVC with BERT / Transformer models

---

## Conclusion

This project demonstrates how NLP and Machine Learning can automate real-world support workflows. By converting raw ticket text into structured categories, the system improves operational efficiency and scalability significantly.
