# Diabetes Prediction using Machine Learning

This project predicts the likelihood of diabetes in patients using the **Pima Indians Diabetes Database**. It includes data exploration, preprocessing, model training, evaluation, and visualization.


---

## Models Implemented

The following machine learning models were implemented and evaluated:
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **Gradient Boosting**

---

## Features

### Data Exploration
- Loaded and explored the dataset.
- Displayed the first few rows, data types, and descriptive statistics.

### Data Preprocessing
- Handled missing values by replacing zeros with medians.
- Scaled features using `StandardScaler`.
- Split data into training (80%) and testing (20%) sets.

### Model Training
- Trained four models: Logistic Regression, SVM, Random Forest, and Gradient Boosting.

### Model Evaluation
- Evaluated models using accuracy, precision, recall, and F1-score.

### Visualization
- Plotted feature distributions, correlation matrix, confusion matrix, ROC curve, and feature importance.

---

## Results

### Model Comparison

| Model               | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|---------------------|----------|---------------------|------------------|--------------------|
| Logistic Regression | 0.7532   | 0.67                | 0.62             | 0.64               |
| SVM                 | 0.7532   | 0.67                | 0.62             | 0.64               |
| Random Forest       | 0.7403   | 0.63                | 0.65             | 0.64               |
| **Gradient Boosting** | **0.7597** | **0.66**            | **0.69**         | **0.67**           |

### Best Model
- **Gradient Boosting** achieved the highest accuracy (75.97%) and best recall (0.69) for the positive class (diabetes).

### Feature Importance
- Visualized feature importance for the Random Forest model, highlighting key predictors like Glucose and BMI.

---

## Outcome

### Best Performing Model
- **Gradient Boosting** was identified as the best-performing model and saved to `diabetes_model.pkl`.

---

## Future Work
- Experiment with hyperparameter tuning for better performance.
- Address class imbalance using techniques like SMOTE.
- Explore deep learning models for comparison.
- Deploy the model as a web application for real-time predictions.

---

## Notes
- The dataset contains medical data for 768 patients.
- Missing values were handled by replacing zeros with medians.
- Feature scaling was applied to ensure consistent model performance.

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

---

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

For questions or feedback, open an issue or contact the author.
