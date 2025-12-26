# Spam Email Detection using Machine Learning

This project implements a machine learning-based system to classify messages/emails as **Spam** or **Not Spam (Ham)** using text processing and classification techniques.

---

## 📌 Project Overview

Spam emails are a common problem that can cause inconvenience and security risks.  
This project uses **Natural Language Processing (NLP)** and **Machine Learning** to automatically detect spam messages based on their textual content.

---

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- TF-IDF Vectorizer  
- Naive Bayes Classifier  
- Jupyter Notebook  

---

## 📂 Project Structure

Spam-Email-Detection/
│
├── spam_detection.py # Python script
├── spam_detection.ipynb # Jupyter Notebook with outputs
├── dataset/
│ └── email_dataset/
│ └── spam.csv # Dataset
├── .gitignore
└── README.md


---

## 📊 Dataset

- **Dataset Name:** SMS Spam Collection Dataset  
- **Labels:**
  - `0` → Not Spam (Ham)
  - `1` → Spam  

The dataset contains text messages labeled as spam or ham.

---

## 🔄 Workflow

1. Load and explore the dataset  
2. Text preprocessing (lowercasing, punctuation removal, stopword removal)  
3. Feature extraction using **TF-IDF**  
4. Split data into training and testing sets  
5. Train a **Naive Bayes** classifier  
6. Evaluate the model using accuracy and classification metrics  
7. Test the model with custom input messages  

---

## ✅ Results

- Achieved **high accuracy (~95–98%)**
- Successfully classified spam and non-spam messages
- Model performs well on unseen data

---

## 🧪 Sample Prediction

```text
Input: "Congratulations! You have won a free prize. Call now."
Output: Spam
