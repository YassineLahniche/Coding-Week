# Obesity Risk Estimation with Explainable ML

A clinical decision-support tool to help physicians accurately estimate obesity risk based on patient lifestyle and physical conditions.

## Project Overview

This project delivers a robust, explainable machine learning solution for obesity risk classification using the UCI ML Repository dataset on estimation of obesity levels. Our solution includes:

- Comprehensive exploratory data analysis
- Implementation and comparison of multiple ML models
- SHAP-based model explainability
- Memory optimization techniques
- An intuitive web interface built with Streamlit

## Key Project Questions

### Dataset Balance Analysis

The dataset contains 7 obesity levels with a relatively balanced distribution:
- Insufficient Weight: 14.1%
- Normal Weight: 14.9%
- Overweight Level I: 14.6%
- Overweight Level II: 13.8%
- Obesity Type I: 14.4%
- Obesity Type II: 14.3%
- Obesity Type III: 13.9%

Although the class distribution is fairly balanced, we implemented class weights in our models to account for the slight variations and ensure optimal performance across all classes.

### Model Performance

We evaluated three advanced tree-based ensemble models:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.92 | 0.91 | 0.92 | 0.91 |
| XGBoost | 0.94 | 0.99 | 0.94 | 0.94 |
| catboost | 0.93 | 0.93 | 0.93 | 0.93 |

**Best Performing Model**: XGBoost Classifier achieved the highest performance across all metrics. We selected it as our final model due to its superior accuracy, precision and recall balance, and its compatibility with SHAP explainability.

### Key Influential Features (SHAP Analysis)

Our SHAP analysis revealed that the following features had the strongest impact on obesity predictions:

1. **Physical Activity Frequency** - moderate negative correlation with higher obesity levels
2. **Food Between Meals** - Frequent snacking moderatly associated with higher obesity risk
3. **Weight** - Direct positive correlation with obesity classification
4. **Age** - strong impact, with middle-aged patients showing higher risk patterns
5. **Caloric dense foods Consumption** - moderate Positive correlation with obesity risk

The SHAP visualizations in our application provide detailed patient-specific explanations for each prediction.

### Prompt Engineering Insights

In our project, we leveraged AI-assisted prompt engineering to enhance our exploratory data analysis (EDA) process, as documented in our PreProcessing.ipynb notebook. This approach significantly accelerated our understanding of the dataset's characteristics and helped us identify critical preprocessing needs. Here, we share our experience with prompt engineering for EDA and the key insights gained.
Our EDA prompt engineering process followed a systematic approach of increasing specificity and technical depth. We began with general, high-level prompts and iteratively refined them to address specific analytical needs and challenges encountered in the dataset.
Initial Exploration Prompt: "Analyze the UCI Obesity Estimation dataset to identify key patterns, distributions, and potential issues for preprocessing."
This initial prompt produced basic statistical summaries but lacked the depth needed for healthcare applications. We enhanced it with specific requirements:
Refined Clinical Context Prompt: "Perform exploratory data analysis on the obesity dataset with a focus on clinical relevance. Identify potential correlations between lifestyle factors and obesity levels, analyze the distribution of each obesity class, and suggest preprocessing steps that preserve medical interpretability."
The refined prompt generated more targeted analyses but still missed some critical data quality assessments. We further improved it by adding specific technical requirements:
Technical EDA Prompt: "Generate Python code to perform comprehensive EDA on the obesity dataset including:

Missing value detection and visualization
Outlier identification using IQR and visualization with boxplots
Feature distribution analysis with respect to obesity classifications
Correlation analysis with heatmaps and statistical significance testing
Class imbalance assessment with detailed metrics
Feature engineering recommendations for health indicators"

This technically specific prompt produced exceptionally valuable results, generating code that:

Identified subtle patterns in how eating habits correlate with obesity levels
Discovered that while no explicit missing values existed, certain anomalous values required treatment
Revealed moderate multicollinearity between physical activity metrics that needed addressing
Showed that our class distribution was more balanced than typical medical datasets, but still benefited from minor class weighting
Identified key feature engineering opportunities by combining related lifestyle indicators

The effectiveness of our prompts significantly improved when we:

Included domain-specific terminology related to medical diagnostics
Requested specific visualization types rather than general analyses
Asked for concrete recommendations rather than just observations
Referenced specific statistical tests appropriate for categorical health data
Specified expected output formats (e.g., Pandas DataFrames, Matplotlib figures)

This prompt engineering experience provided valuable insights beyond the immediate EDA tasks. We found that well-crafted prompts not only accelerated our analysis but also ensured we addressed critical clinical considerations that might otherwise have been overlooked. The process helped bridge the gap between statistical analysis and medical domain knowledge, ultimately improving the quality and interpretability of our obesity risk prediction model.

## Repository Structure

```
Coding-Week/
├── .github/
│   └── workflows/
│       ├── model_test.yml        # CI workflow for model testing
│       └── ui_test.yml           # CI workflow for UI testing
├── data/
│   ├── data.csv                  # Raw dataset file
│   └── processed_obesity_dataset.csv # Preprocessed dataset
├── model/
│   ├── catboost_model.pkl        # Saved CatBoost model
│   ├── rf_model.pkl              # Saved Random Forest model
│   ├── xgb_baseline.pkl          # Saved XGBoost baseline model
│   └── xgb_tuned_.pkl            # Saved tuned XGBoost model
├── utils/
│   │   ├── Preprocessing.py      # Data preprocessing utilities
│   │   └── evaluation.py         # Model evaluation functions
├── notebooks/
│   ├── PreProcessing.ipynb       # Data preprocessing notebook
│   ├── Training_models.ipynb     # Model training notebook
│   └── memory_optimization.ipynb # Memory optimization experiments
├── tests/
│   ├── test_model.py             # Unit tests for model
│   └── test_ui.py                # Unit tests for UI
├── app.py                        # Streamlit application
├── main.py                       # Main application entry point
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```


## Development Team
- Yassine LAHNICHE 
- Youness TOURABI
- Othmane TEMRI
- Abdellah OUAZZANI TAYBI

## Huge Thank You
- Kawtar ZERHOUNI
- Hermann Leibniz Klauss Agossou

