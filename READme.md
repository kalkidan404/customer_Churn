# Customer Churn Prediction

## Project Overview

Customer churn occurs when customers stop using a company's products or services. For telecommunications companies, identifying customers who are likely to leave can help businesses develop targeted retention strategies and improve customer relationships.

This project uses machine learning to predict whether a telecommunications customer is likely to churn based on their account information, service subscriptions, contract details, and billing information.

The goal is to build a classification model that helps identify potential churners so that businesses can take proactive steps to retain them.

## Objectives

- Clean and prepare customer data for machine learning.
- Perform exploratory data analysis (EDA) to investigate patterns associated with churn.
- Preprocess numerical and categorical features.
- Train and compare multiple classification models.
- Evaluate model performance using appropriate classification metrics.
- Adjust the classification threshold to improve churn detection.
- Build and save a reusable machine learning pipeline.

## Dataset

This project uses the **Telco Customer Churn dataset** from IBM's sample datasets.

The dataset contains customer information, including:

- **Customer demographics:** Gender, senior citizen status, partner status, and dependents.
- **Account information:** Tenure, contract type, and billing preferences.
- **Services:** Phone service, internet service, online security, technical support, and other subscriptions.
- **Billing:** Monthly charges and total charges.
- **Target variable:** Whether the customer churned.

The original dataset contains 7,043 customer records and 21 columns.

During data cleaning, 11 records with missing `TotalCharges` values were removed, leaving **7,032 records** for analysis.

The `customerID` column was excluded from model training because it serves as a customer identifier rather than a meaningful predictive feature.

The target variable, `Churn`, was encoded as:

- `0` — No churn
- `1` — Churn

The dataset is not included in this repository because the raw data is kept out of version control.

## Technologies Used

- **Python** — Programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical operations
- **Matplotlib / Seaborn** — Data visualization, where applicable
- **Scikit-learn** — Data preprocessing, model training, evaluation, and pipelines
- **XGBoost** — Gradient-boosted tree classification
- **Joblib** — Saving and loading the trained model
- **Jupyter Notebook** — Interactive development and experimentation
- **Git and GitHub** — Version control and project hosting

## Project Workflow

### 1. Data Cleaning

The dataset was inspected for missing values, data types, and potential data quality issues.

The `TotalCharges` column initially contained blank strings despite appearing to have no missing values according to the initial null-value check.

The column was converted to a numeric data type, the resulting missing values were identified, and the affected records were removed.

After cleaning, the dataset contained 7,032 records with no missing values.

### 2. Exploratory Data Analysis (EDA)

EDA was performed to investigate relationships between customer characteristics and churn.

Key observations included:

- Customers with month-to-month contracts had a substantially higher churn rate than customers with longer-term contracts.
- Customers who churned had lower average tenure than customers who stayed.
- Customers who churned had higher average monthly charges than customers who stayed.

These observations indicate associations in the dataset; they do not establish that any individual feature causes churn.

### 3. Feature Preparation

The target variable was separated from the input features.

The `customerID` column was excluded, leaving 19 input features.

The dataset was split into training and testing sets using an 80/20 split with stratification to preserve the class distribution.

A separate validation set was created from the training data for model selection and experimentation.

### 4. Data Preprocessing

A preprocessing pipeline was created using Scikit-learn's `ColumnTransformer`.

- **StandardScaler:** Standardized numerical features.
- **OneHotEncoder:** Converted categorical features into numerical representations.
- **handle_unknown="ignore":** Allowed the encoder to handle previously unseen categories during prediction.

The preprocessing steps were fitted on training data rather than the validation or test data to reduce data leakage.

### 5. Model Training and Comparison

Several classification algorithms were explored:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

The models were evaluated using accuracy, precision, recall, and F1-score.

Because the dataset contains more customers who stayed than customers who churned, accuracy alone was not sufficient to assess model performance.

Particular attention was given to the metrics for the churn class.

### 6. Threshold Selection

The default classification threshold of 0.50 was explored alongside alternative thresholds.

A threshold of **0.40** was selected for the final saved configuration because it offered a reasonable balance between identifying churners and avoiding unnecessary customer contacts.

At this threshold, customers with a predicted churn probability of at least 40% are classified as potential churners.

### 7. Pipeline and Model Persistence

A Scikit-learn `Pipeline` was created to combine preprocessing and Logistic Regression into one reusable object.

The fitted pipeline and classification threshold were saved using Joblib.

The saved files were subsequently loaded to verify that the pipeline could generate predictions without being trained again.

## Model Performance

The selected configuration is Logistic Regression with a classification threshold of 0.40.

The following results were obtained on the validation set:

| Metric          |  Score |
| --------------- | -----: |
| Accuracy        | 77.96% |
| Churn Precision |    58% |
| Churn Recall    |    64% |
| Churn F1-score  |    61% |

### Interpreting the Results

**Accuracy — 77.96%**

Approximately 78% of validation predictions were correct.

**Precision — 58%**

Of the customers predicted to churn, 58% actually churned.

**Recall — 64%**

The model identified 64% of the customers who actually churned.

**F1-score — 61%**

The F1-score balances precision and recall, providing a useful measure of the model's ability to identify churners.

### Business Interpretation

The selected threshold prioritizes identifying potential churners rather than maximizing accuracy alone.

For a telecommunications company, contacting some customers who would have stayed anyway may be acceptable if the cost of those contacts is lower than the expected benefit of retaining customers who might otherwise leave.

However, the financial value of this approach depends on factors such as retention campaign costs, customer lifetime value, and the effectiveness of retention offers.

The selected threshold is a practical modeling choice, not a proven financial optimum.

**Note:** These figures are validation results. A final evaluation on the held-out test set has not yet been completed as part of this workflow.

## Project Structure

The repository is organized around the following files:

```text
customer_Churn/
│
├── raw_data/
│   └── Telco-Customer-Churn.csv
│
├── churn_model.pkl
├── churn_threshold.pkl
├── README.md
└── requirements.txt
```

The notebook used for exploration and training should also be included in the repository. Add its actual filename to this structure when documenting your final project files.

The raw dataset is shown for reference but is not intended to be committed to Git.

- `churn_model.pkl` — Saved preprocessing and Logistic Regression pipeline.
- `churn_threshold.pkl` — Saved classification threshold of 0.40.
- `README.md` — Project documentation.
- `requirements.txt` — Python dependencies.

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd customer_Churn
```

Replace `<your-repository-url>` with the URL of your GitHub repository.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

## Loading the Saved Model

After installing the dependencies and obtaining the saved model files, the pipeline can be loaded with Joblib.

```python
import joblib

# Load the trained pipeline and threshold
model = joblib.load("churn_model.pkl")
threshold = joblib.load("churn_threshold.pkl")

print("Model loaded successfully!")
print("Classification threshold:", threshold)
```

To predict churn for new customers, provide a Pandas DataFrame containing the same 19 input features used during training.

```python
# new_customers must contain the expected input features
probabilities = model.predict_proba(new_customers)[:, 1]

predictions = (probabilities >= threshold).astype(int)

print("Churn probabilities:", probabilities)
print("Churn predictions:", predictions)
```

The pipeline automatically applies the fitted preprocessing steps before generating predictions.

The input DataFrame must use the expected feature names and compatible data types. It must not include the target column, `Churn`, or the excluded identifier column, `customerID`.

Prediction output:

- `0` — Not predicted to churn
- `1` — Predicted to churn

The predicted probabilities are model estimates, not guarantees that a customer will or will not churn.

## Limitations and Future Improvements

Although the model demonstrates how machine learning can support churn prediction, several improvements remain possible:

- Evaluate the selected configuration on the held-out test set.
- Explore cross-validation to obtain more robust performance estimates.
- Investigate probability calibration.
- Evaluate feature importance and model interpretability.
- Test different classification thresholds against realistic business costs.
- Develop a simple interface for making predictions on new customers.
- Investigate how the model performs on data from different periods or customer populations.

The selected threshold and reported metrics should be reassessed if the model is deployed in a real business environment.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for customer churn prediction, from data cleaning and exploratory analysis to preprocessing, model comparison, threshold selection, and model persistence.

Logistic Regression was selected as the final model because it offers a relatively interpretable approach while providing a practical balance between churn precision and recall.

The project provides a foundation for further work in classification, model deployment, and data-driven customer retention strategies.
