# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import seaborn as sns  # For data visualization
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.svm import SVC  # Support Vector Machine model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Ensemble models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay  # For model evaluation
import joblib  # For saving and loading models

# Step 1: Load and explore the dataset
df = pd.read_csv("diabetes.csv")  # Load the dataset

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Display information about the dataset (column names, data types, non-null counts)
print("\nDataset information:")
print(df.info())

# Display descriptive statistics (mean, std, min, max, etc.)
print("\nDescriptive statistics:")
print(df.describe())

# Step 2: Data preprocessing
# Handle missing values (replace zeros with median values for specific columns)
columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer = SimpleImputer(missing_values=0, strategy='median')  # Initialize the imputer
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])  # Apply imputation

# Split the data into features (X) and target variable (y)
X = df.drop('Outcome', axis=1)  # Features (all columns except 'Outcome')
y = df['Outcome']  # Target variable ('Outcome')

# Scale the features using StandardScaler (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit and transform the features

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  # 80% training, 20% testing

# Step 4: Train and evaluate models
# Define a dictionary of models to compare
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Train each model, make predictions, and evaluate performance
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Make predictions on the test set
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "classification_report": classification_report(y_test, y_pred),  # Detailed classification metrics
        "confusion_matrix": confusion_matrix(y_test, y_pred)  # Confusion matrix
    }
    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{results[name]['classification_report']}")

# Step 5: Visualizations
# Plot the distribution of each feature
plt.figure(figsize=(12, 8))
for i, column in enumerate(X.columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[column], kde=True, bins=30)  # Histogram with KDE
    plt.title(column)
plt.tight_layout()
plt.show()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()  # Calculate correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')  # Heatmap of correlations
plt.title("Correlation Matrix")
plt.show()

# Confusion matrix for the best model (Random Forest)
best_model = results["Random Forest"]["model"]  # Select the best model
plt.figure(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(y_test, best_model.predict(X_test), cmap='Blues', display_labels=['No Diabetes', 'Diabetes'])
plt.title("Confusion Matrix (Random Forest)")
plt.show()

# ROC curve for the best model
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(best_model, X_test, y_test)  # Plot ROC curve
plt.title("ROC Curve (Random Forest)")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line for random model
plt.show()

# Compare model accuracy using a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=[results[name]["accuracy"] for name in results], palette='viridis')
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.7, 0.9)  # Set y-axis limits
plt.show()

# Feature importance for Random Forest
feature_importances = best_model.feature_importances_  # Get feature importances
feature_names = X.columns  # Feature names
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names, palette='viridis')  # Bar plot of feature importances
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Step 6: Save the best model
best_model_name = "Gradient Boosting"  # Best model based on evaluation
best_model = results[best_model_name]["model"]  # Get the best model

# Save the model to a file
joblib.dump(best_model, 'diabetes_model.pkl')
print(f"\nBest model ({best_model_name}) saved to 'diabetes_model.pkl'.")

# Step 7: Example of using the saved model
loaded_model = joblib.load('diabetes_model.pkl')  # Load the saved model

# New data for prediction (as a list of lists)
new_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]

# Convert new data to a DataFrame with the same feature names
new_data_df = pd.DataFrame(new_data, columns=X.columns)

# Scale the new data using the trained StandardScaler
new_data_scaled = scaler.transform(new_data_df)

# Make a prediction
prediction = loaded_model.predict(new_data_scaled)
print(f"\nPrediction for new data: {'Diabetes' if prediction[0] == 1 else 'No Diabetes'}")
