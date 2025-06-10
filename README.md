# 🔬 Breast Cancer Classification using Random Forest (Script + Streamlit Ready)

This project provides a complete end-to-end machine learning solution for classifying breast cancer tumors as **malignant** or **benign** using the **Breast Cancer Wisconsin dataset** from `sklearn`. It includes both a Python script for EDA and model building, as well as an optional Streamlit interface for interactive predictions.

---

## 🚀 Project Description

The notebook and script:

* Load and preprocess the breast cancer dataset
* Visualize feature correlations and class distributions
* Train a **Random Forest Classifier** with `scikit-learn`
* Evaluate performance using metrics like Confusion Matrix, Classification Report, ROC Curve, and AUC Score
* Rank feature importance and visualize top predictors

Additionally, the project can be enhanced with a `streamlit_app.py` to allow real-time user interaction with the model via a web interface.

---

## 📊 Key Features

* ✅ Built-in dataset (no file uploads needed)
* ✅ Clean preprocessing using StandardScaler
* ✅ Model training with RandomForestClassifier
* ✅ Visualizations: Heatmap, Confusion Matrix, ROC Curve, Feature Importance
* ✅ Evaluation metrics: AUC, Accuracy, Precision, Recall, F1-score
* ✅ Ready for Streamlit integration

---

## 🧪 Libraries Used

* `pandas`, `numpy`
* `matplotlib`, `seaborn`
* `scikit-learn`
* (Optional) `streamlit`

---

## 📂 Project Structure

| File                            | Description                                                      |
| ------------------------------- | ---------------------------------------------------------------- |
| `breast_cancer_rf.py`           | Main script with data processing, model training, and evaluation |
| `streamlit_app.py` *(optional)* | Streamlit version for interactive web interface                  |
| `README.md`                     | Project overview and usage guide                                 |

---

## ⚙️ How to Run

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/breast-cancer-classification.git
cd breast-cancer-classification
```

2. **Install dependencies**:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

3. **Run the script**:

```bash
python breast_cancer_rf.py
```

4. *(Optional)* **Launch the Streamlit app**:

```bash
streamlit run streamlit_app.py
```

---

## 📈 Output Visuals

* 📌 Count plot of benign vs malignant tumors
* 🔥 Correlation heatmap
* ✅ Confusion matrix of model performance
* 🎯 ROC Curve with AUC Score
* 🌟 Top 10 important features in prediction

---

## 👨‍⚕️ Dataset Info

* **Source**: `sklearn.datasets.load_breast_cancer()`
* **Features**: 30 numerical features
* **Target**: `0 = Malignant`, `1 = Benign`

---

## 👤 Author

**Shreesha Shetty**
*ML Enthusiast | Data Science Student*

---

## 📌 License

This project is open-source and licensed under the **MIT License**.
