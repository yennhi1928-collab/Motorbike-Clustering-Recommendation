# 🚀 Project 2 – Motorbike Clustering & Recommendation  
### **User Guide (Streamlit Application)**

This document explains how to run and use the **Motorbike Clustering & Recommendation** Streamlit application.  
The system includes **content-based recommendation** and **clustering** models built from motorbike data.

---

## 📌 Environment & Requirements
- **Python Version:** 3.10  
- **Streamlit App Folder:** `Motorbike Clustering & Recommendation`  
- **Recommended Tools:** Streamlit, GitHub  

---

## 📂 Data Files & Resources

### **Main Dataset**
- `data_motobikes.xlsx`

### **Supporting NLP Files** (located in the `files/` folder)
- `emojicon.txt`  
- `english-vnmese.txt`  
- `teencode.txt`  
- `vietnamese-stopwords.txt`  
- `wrong-word.txt`

### **Model Files**
- `xe_gemsim_sim.pkl` – Gensim similarity model  
- `cluster_pipeline.pkl` – Preprocessing + clustering pipeline  

### **Core Function**
#### `process_query(text)`
Applies the following NLP pipeline:
- Unicode normalization & lowercase  
- Emoji replacement  
- Teencode processing  
- Wrong-word correction  
- Text cleaning  
- English → Vietnamese translation  
- Remove punctuation  
- Tokenization  
- POS tagging  
- Stopword removal  

**Output:** a fully processed & tokenized text string.

---

# 🖥️ GUI Overview (4 Tabs)

## **1️⃣ Introduction**
Displays:
- Project overview  
- Number of samples  
- Summary of clusters  

---

## **2️⃣ Model Selection**
Explains:
- Data preprocessing workflow  
- Experiments with multiple ML algorithms  
- Metric-based model comparison  
- Selection of the best-performing scikit-learn model  
- Team information  

---

## **3️⃣ Motorbike Recommendation**
Allows users to:
- Select a motorbike from a dropdown **or**
- Enter a text description  

Click **“Find Similar Motorbikes”** to display the **top 3 most similar motorbikes** using cosine similarity.

---

## **4️⃣ Motorbike Clustering**
Input parameters:
- Brand  
- Year of Registration  
- Engine Capacity  
- Type  
- Kilometers Travelled  
- Origin  
- Price (million VND)  

Click **“Predict Cluster”** to determine the cluster that best matches the motorbike.

---

## 🔒 License
**For internal use only.**  
Developed by our team as part of the **Data Science & Machine Learning Certification – Project 2: Motorbike Clustering & Recommendation**.

---
