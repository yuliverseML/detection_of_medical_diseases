# Diabetes Prediction Model

A machine learning system for predicting diabetes risk using various clinical indicators. This project implements and compares multiple ML algorithms to identify the most effective approach for diabetes prediction.

This project uses the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) from Kaggle.

## Models Implemented

The project compares several machine learning algorithms:

- **Logistic Regression**: A linear model with L1/L2 regularization
- **Random Forest**: An ensemble of decision trees with bagging
- **Gradient Boosting**: An ensemble method that builds trees sequentially
- **XGBoost**: An optimized gradient boosting implementation
- **Ensemble Model**: A voting classifier combining all models above

## Features

### Data Exploration

- Comprehensive initial data analysis
- Missing value identification 
- Class distribution analysis
- Statistical summary of features
- Medical feature validation

### Data Preprocessing

- KNN imputation for missing values
- Feature engineering (BMI categories, glucose-insulin ratio, age groups)
- Outlier treatment using winsorization
- Yeo-Johnson power transformation for skewed features
- Train-test split with stratification

### Model Training

- Hyperparameter tuning via GridSearchCV
- Cross-validation with stratified k-fold
- Class imbalance handling
- Ensemble model creation with soft voting

### Model Evaluation

- Multiple performance metrics:
  - Accuracy, Precision, Recall
  - F1 Score, ROC AUC
- Classification reports
- Confidence intervals
- Risk stratification

### Visualization

- Feature distribution analysis by outcome class
- Correlation heatmap with diverging color palette
- ROC curves for all models
- Precision-Recall curves
- Confusion matrices
- Feature importance plots
- Model comparison bar charts

## Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7662 | 0.6250 | 0.8333 | 0.7143 | 0.8380 |
| Random Forest | 0.7273 | 0.6200 | 0.5741 | 0.5962 | 0.8094 |
| Gradient Boosting | 0.7403 | 0.6458 | 0.5741 | 0.6078 | 0.8165 |
| XGBoost | 0.7532 | 0.6667 | 0.5926 | 0.6275 | 0.8181 |
| Ensemble | 0.7338 | 0.6226 | 0.6111 | 0.6168 | 0.8235 |

### Best Model

**Logistic Regression** achieved the highest ROC AUC (0.8380) and recall (0.8333), making it the most effective model for this task. Its high recall is particularly valuable in medical screening scenarios where missing positive cases (false negatives) is more costly than false positives.

## Outcome

### Best Performing Model

The **Logistic Regression** model was selected for deployment with the following characteristics:

- Strongest overall performance with ROC AUC of 0.8380
- High recall (83.33%) for detecting diabetes cases
- Good precision-recall balance (F1 score: 0.7143)
- Easily interpretable coefficients for clinical context
- Deployed as a production-ready prediction function

For an example patient with elevated glucose (148), BMI of 33.6, and age 50, the model predicts:
- **Prediction**: Positive for diabetes risk
- **Probability**: 74.65%
- **Risk Level**: High

## 🚀 Future Work

- **Additional Features**: Incorporate HbA1c, cholesterol, and lifestyle factors
- **Model Improvement**: Explore neural networks and deep learning approaches
- **Temporal Analysis**: Add longitudinal data to predict diabetes development over time
- **Explainability**: Implement SHAP values for more detailed prediction explanations
- **Mobile Integration**: Develop a mobile app for patient self-assessment
- **Validation**: Perform external validation on diverse patient populations
- **Risk Calibration**: Calibrate risk scores to specific clinical guidelines

## 📝 Notes

- The code includes comprehensive error handling and data validation
- All visualizations use a consistent color scheme for readability
- The system is designed to err on the side of caution (higher recall)
- Model deployment includes proper versioning and reproducibility measures

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<p align="center">
  Made with ❤️ for better diabetes prediction
</p>
