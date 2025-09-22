<img width="1919" height="1079" alt="image" src="<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/a750e4b3-efea-4b4b-805c-b5906f4b036a" />
" /># 🏦 Bank Term Deposit Subscription Prediction



This project delivers an end-to-end machine learning solution for predicting whether a bank client will subscribe to a term deposit. By leveraging advanced modeling techniques, this predictor enables targeted marketing campaigns, optimizes resource allocation, and provides data-driven insights for strategic decision-making.

## Live: https://banktermdepositsubscriptionprediction.streamlit.app/
---
## ✨ Key Features & Technical Highlights

-   **End-to-End ML Pipeline**: A complete, reproducible workflow from raw data ingestion to a saved, production-ready model using `scikit-learn` and `imblearn` pipelines.
-   **Advanced Feature Engineering**: Creation of interaction features (e.g., `balance_per_age`) and meaningful ordinal encoding for features like `education`.
-   **Robust Imbalance Handling**: Utilization of `SMOTETomek` to intelligently oversample the minority class and clean the decision boundary, leading to a more balanced and high-performing model.
-   **Comprehensive Model Evaluation**: Benchmarking of baseline models against advanced gradient boosting models (LightGBM, XGBoost, CatBoost).
-   **Sophisticated Ensembling**: Implementation of a **Stacking Classifier** that uses the tuned gradient boosting models as base learners to achieve superior and more robust performance.
-   **GPU Acceleration**: The entire tuning and training process is optimized to run on a GPU, drastically reducing computation time from hours to minutes.
-   **Model Explainability**: In-depth model interpretation using **SHAP (SHapley Additive exPlanations)** to understand the key drivers behind the final model's predictions.
-   **Ready for Deployment**: The project concludes with a saved, unified pipeline and a complete Streamlit application script (`app.py`) for live predictions.

---
## 🛠️ Tech Stack

| Category                 | Technologies & Libraries                                                                              |
| ------------------------ | ----------------------------------------------------------------------------------------------------- |
| **Data Manipulation** | `Python`, `Pandas`, `NumPy`                                                                           |
| **Data Visualization** | `Matplotlib`, `Seaborn`                                                                               |
| **ML Pipeline & Modeling** | `Scikit-learn`, `Imbalanced-learn`, `LightGBM`, `XGBoost`, `CatBoost`                                   |
| **Explainability** | `SHAP`                                                                                                |
| **Deployment** | `Streamlit`, `Joblib`                                                                                 |
| **Development** | `Jupyter Notebook`, `Kaggle Notebooks`, `Google Colab`                                                  |

---
## 🔄 Project Workflow

The project follows a structured workflow designed to ensure robustness and prevent data leakage:

`Data Loading` → `Feature Engineering` → `Data Splitting (80/20)` → `Preprocessing Pipeline (Fit on Train)` → `Resampling (on Train)` → `Hyperparameter Tuning (CV on Resampled Train)` → `Stacking Ensemble` → `Final Evaluation` → `SHAP Analysis` → `Save Artifacts`

---
## 📊 Final Model Performance

The models were evaluated on the 20% held-out test set after optimizing the prediction threshold to maximize the F1-Score. The leaderboard below shows that the tuned **LightGBM** model achieved the highest F1-Score in this run, closely followed by the other advanced models.

| Model               | Accuracy | Precision (Yes) | Recall (Yes) | F1-Score (Yes) | ROC-AUC |
| ------------------- | :------: | :-------------: | :----------: | :------------: | :-----: |
| **LightGBM** |  0.9325  |     0.6949      |    0.7856    |   **0.7375** | 0.9674  |
| **XGBoost** |  0.9292  |     0.6786      |    0.7844    |     0.7277     | 0.9650  |
| **CatBoost** |  0.9280  |     0.6701      |    0.7940    |     0.7268     | 0.9648  |
| **Stacking Ensemble** |  0.9286  |     0.6757      |    0.7853    |     0.7264     | 0.9628  |

---
## 🧠 Model Explainability with SHAP

The SHAP summary plot reveals the most influential features driving the model's predictions for the positive class ("Yes" - will subscribe).



**Key Insights:**
-   **`duration`**: The duration of the last contact is by far the most significant predictor. Longer calls are strongly associated with a higher likelihood of subscription.
-   **`poutcome_success`**: A successful outcome in a previous campaign is a very strong positive indicator.
-   **Financials & Demographics**: Features like `balance`, `age`, and `housing` also play important roles, with their impact varying for each individual prediction.

---
## 📁 Repository Structure
```
bank-term-deposit-prediction/
│
├── 📂 data/
│   └── train.csv
│
├── 📂 saved_models/
│   ├── stacking_model_final_gpu.joblib
│   └── preprocessor_final_gpu.joblib
│
├── 📜 Bank.ipynb
├── 📜 Model.ipynb  
├── 📜 app.py                  
├── 📜 requirements.txt      
└── 📜 README.md
```

---
## ⚙️ Setup and Installation

Follow these steps to set up the project environment.

**1. Clone the Repository**
```bash
git clone [https://github.com/itz-Mayank/Bank_Term_Deposit_Subscription_Prediction](https://github.com/itz-Mayank/Bank_Term_Deposit_Subscription_Prediction)]
cd Bank_Term_Deposit_Subscription_Prediction
```
**2. Create a Python Environment**

It is highly recommended to use a virtual environment (e.g., Conda) to manage dependencies.

```
conda create -n bank_predictor python=3.10 -y
conda activate bank_predictor
```

**3. Install Dependencies**

Install all the required libraries from the requirements.txt file.
```
pip install -r requirements.txt
```
### 🚀 Usage

There are two main ways to use this repository:

**1. Training the Model**

To re-run the entire training, tuning, and evaluation process, open and run the main notebook:
```
jupyter lab main_notebook.ipynb
```

**2. Running the Prediction App**
To launch the user-facing web application for live predictions, ensure you have the saved model files (e.g., final_model_pipeline.joblib) in the root directory and run:
```
streamlit run app.py
```
✍️ Author
Created by `Mayank Meghwal`
