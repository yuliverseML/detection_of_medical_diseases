# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ML preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFE

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ML evaluation
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, roc_curve, precision_recall_curve, 
                            confusion_matrix, classification_report, 
                            ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)

# For model persistence
import joblib

# Configure visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def load_and_explore_data(filepath):
    """
    Load dataset and perform initial exploration
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        The loaded dataframe
    """
    print("="*50)
    print("LOADING AND EXPLORING DATASET")
    print("="*50)
    
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Display basic information
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    print("\nDataset shape:", df.shape)
    
    print("\nDataset information:")
    print(df.info())
    
    print("\nDescriptive statistics:")
    print(df.describe().round(2))
    
    print("\nMissing values (including zeros in medical features):")
    # Count zeros as missing values for certain medical measurements where 0 is physiologically impossible
    medical_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    zeros = (df[medical_features] == 0).sum()
    print(zeros)
    
    # Check class distribution
    print("\nClass distribution:")
    class_counts = df['Outcome'].value_counts()
    print(class_counts)
    print(f"Percentage of diabetic cases: {class_counts[1]/len(df):.2%}")
    
    return df

# Load dataset
df = load_and_explore_data("diabetes.csv")

def visualize_data(df):
    """
    Create comprehensive data visualizations for exploratory data analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to visualize
    """
    print("="*50)
    print("DATA VISUALIZATION")
    print("="*50)
    
    # 1. Class distribution visualization
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='Outcome', data=df, palette=['#3498db', '#e74c3c'])
    plt.title('Class Distribution (Diabetes Outcome)', fontsize=15)
    plt.xlabel('Diabetes Diagnosis (0=No, 1=Yes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add count labels
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width()/2., p.get_height()),
                   ha='center', va='bottom', fontsize=12)
    
    # Add percentage labels
    total = len(df)
    for i, p in enumerate(ax.patches):
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax.annotate(percentage, 
                   (p.get_x() + p.get_width()/2., p.get_height()/2),
                   ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    plt.show()
    
    # 2. Feature distributions by outcome
    plt.figure(figsize=(16, 12))
    for i, feature in enumerate(df.drop('Outcome', axis=1).columns):
        plt.subplot(3, 3, i+1)
        for outcome in [0, 1]:
            sns.kdeplot(df.loc[df['Outcome'] == outcome, feature], 
                        label=f'Outcome {outcome}', fill=True, alpha=0.5)
        plt.title(f'Distribution of {feature} by Outcome', fontsize=12)
        plt.legend(title='Diabetes')
    plt.tight_layout()
    plt.show()
    
    # 3. Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr().round(2)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, annot=True, mask=mask, cmap=cmap, vmax=1, vmin=-1, 
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Matrix', fontsize=15)
    plt.show()
    
    # 4. Pairplot for key features
    key_features = ['Glucose', 'BMI', 'Age', 'Insulin', 'Outcome']
    sns.pairplot(df[key_features], hue='Outcome', palette=['#3498db', '#e74c3c'], 
                 diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'})
    plt.suptitle('Pairplot of Key Features', y=1.02, fontsize=16)
    plt.show()
    
    # 5. Boxplots to identify outliers
    plt.figure(figsize=(16, 12))
    for i, feature in enumerate(df.drop('Outcome', axis=1).columns):
        plt.subplot(3, 3, i+1)
        sns.boxplot(x='Outcome', y=feature, data=df, palette=['#3498db', '#e74c3c'])
        plt.title(f'Boxplot of {feature} by Outcome', fontsize=12)
    plt.tight_layout()
    plt.show()

# Visualize data
visualize_data(df)

def preprocess_data(df):
    """
    Perform data preprocessing including handling missing values, 
    outlier treatment, and feature scaling
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataframe to preprocess
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, scaler, features_list)
    """
    print("="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # 1. Handle missing values (zeros in medical features)
    medical_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    print("\nReplacing physiologically impossible zeros with NaN...")
    for feature in medical_features:
        # Replace 0 with NaN
        data.loc[data[feature] == 0, feature] = np.nan
    
    # Use KNN imputation for missing values
    print("Applying KNN imputation for missing values...")
    imputer = KNNImputer(n_neighbors=5)
    data.loc[:, data.columns != 'Outcome'] = imputer.fit_transform(data.loc[:, data.columns != 'Outcome'])
    
    # 2. Feature engineering
    print("\nPerforming feature engineering...")
    # BMI categories
    data['BMI_Category'] = pd.cut(data['BMI'], 
                                  bins=[0, 18.5, 24.9, 29.9, 100],
                                  labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    data = pd.get_dummies(data, columns=['BMI_Category'], drop_first=True)
    
    # Glucose-Insulin interaction
    data['Glucose_Insulin_Ratio'] = data['Glucose'] / (data['Insulin'] + 1)  # Add 1 to avoid division by zero
    
    # Age groups
    data['Age_Group'] = pd.cut(data['Age'], bins=[20, 30, 40, 50, 100], 
                               labels=['20-30', '30-40', '40-50', '50+'])
    data = pd.get_dummies(data, columns=['Age_Group'], drop_first=True)
    
    # 3. Outlier treatment using winsorization
    print("\nHandling outliers with winsorization...")
    for feature in data.select_dtypes(include=['float64', 'int64']).columns:
        if feature != 'Outcome':
            # Apply 5th and 95th percentile winsorization
            lower_bound, upper_bound = np.percentile(data[feature], [5, 95])
            data[feature] = data[feature].clip(lower_bound, upper_bound)
    
    # 4. Split data
    print("\nSplitting data into training and testing sets...")
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    # Store feature names for later use
    features_list = X.columns.tolist()
    
    # 5. Feature scaling
    print("Applying feature scaling...")
    # Use Yeo-Johnson power transform for skewed features
    scaler = PowerTransformer(method='yeo-johnson')
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\nFinal training set shape: {X_train.shape}")
    print(f"Final testing set shape: {X_test.shape}")
    print(f"Feature engineering added {len(features_list) - 8} new features")
    
    return X_train, X_test, y_train, y_test, scaler, features_list

# Preprocess data
X_train, X_test, y_train, y_test, scaler, features = preprocess_data(df)

def train_and_evaluate_models(X_train, X_test, y_train, y_test, features):
    """
    Train multiple models, tune hyperparameters, and evaluate performance
    
    Parameters:
    -----------
    X_train, X_test : numpy.ndarray
        Training and testing feature sets
    y_train, y_test : pandas.Series
        Training and testing target variables
    features : list
        List of feature names
        
    Returns:
    --------
    dict
        Dictionary containing trained models and their evaluation metrics
    """
    print("="*50)
    print("MODEL TRAINING AND EVALUATION")
    print("="*50)
    
    # Initialize results dictionary
    results = {}
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. Logistic Regression with hyperparameter tuning
    print("\nTraining Logistic Regression model...")
    lr_params = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': ['balanced', None]
    }
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_grid = GridSearchCV(lr, lr_params, cv=cv, scoring='roc_auc', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    
    lr_best = lr_grid.best_estimator_
    lr_pred = lr_best.predict(X_test)
    lr_pred_proba = lr_best.predict_proba(X_test)[:, 1]
    
    print(f"Best Logistic Regression parameters: {lr_grid.best_params_}")
    
    # 2. Random Forest with hyperparameter tuning
    print("\nTraining Random Forest model...")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=cv, scoring='roc_auc', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    
    rf_best = rf_grid.best_estimator_
    rf_pred = rf_best.predict(X_test)
    rf_pred_proba = rf_best.predict_proba(X_test)[:, 1]
    
    print(f"Best Random Forest parameters: {rf_grid.best_params_}")
    
    # 3. Gradient Boosting with hyperparameter tuning
    print("\nTraining Gradient Boosting model...")
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=cv, scoring='roc_auc', n_jobs=-1)
    gb_grid.fit(X_train, y_train)
    
    gb_best = gb_grid.best_estimator_
    gb_pred = gb_best.predict(X_test)
    gb_pred_proba = gb_best.predict_proba(X_test)[:, 1]
    
    print(f"Best Gradient Boosting parameters: {gb_grid.best_params_}")
    
    # 4. XGBoost with hyperparameter tuning
    print("\nTraining XGBoost model...")
    xgb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_grid = GridSearchCV(xgb, xgb_params, cv=cv, scoring='roc_auc', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    
    xgb_best = xgb_grid.best_estimator_
    xgb_pred = xgb_best.predict(X_test)
    xgb_pred_proba = xgb_best.predict_proba(X_test)[:, 1]
    
    print(f"Best XGBoost parameters: {xgb_grid.best_params_}")
    
    # 5. Create an ensemble (Voting Classifier)
    print("\nTraining Ensemble model (Voting Classifier)...")
    
    ensemble = VotingClassifier(
        estimators=[
            ('lr', lr_best),
            ('rf', rf_best),
            ('gb', gb_best),
            ('xgb', xgb_best)
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    
    ensemble_pred = ensemble.predict(X_test)
    ensemble_pred_proba = ensemble.predict_proba(X_test)[:, 1]
    
    # Store all models and their predictions
    models = {
        "Logistic Regression": {
            "model": lr_best,
            "predictions": lr_pred,
            "probabilities": lr_pred_proba
        },
        "Random Forest": {
            "model": rf_best,
            "predictions": rf_pred,
            "probabilities": rf_pred_proba
        },
        "Gradient Boosting": {
            "model": gb_best,
            "predictions": gb_pred,
            "probabilities": gb_pred_proba
        },
        "XGBoost": {
            "model": xgb_best,
            "predictions": xgb_pred,
            "probabilities": xgb_pred_proba
        },
        "Ensemble": {
            "model": ensemble,
            "predictions": ensemble_pred,
            "probabilities": ensemble_pred_proba
        }
    }
    
    # Calculate and store evaluation metrics for each model
    print("\nEvaluating all models...")
    for name, model_dict in models.items():
        y_pred = model_dict["predictions"]
        y_proba = model_dict["probabilities"]
        
        # Classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # Store metrics
        models[name]["metrics"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc
        }
        
        # Print metrics
        print(f"\n{name} Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC: {auc:.4f}")
        print(f"  Classification Report:\n{classification_report(y_test, y_pred)}")
        
    # Determine the best model based on ROC AUC
    best_model_name = max(models, key=lambda x: models[x]["metrics"]["roc_auc"])
    print(f"\nBest model based on ROC AUC: {best_model_name} with AUC {models[best_model_name]['metrics']['roc_auc']:.4f}")
    
    # Feature importance for the best model (if applicable)
    best_model = models[best_model_name]["model"]
    if hasattr(best_model, 'feature_importances_'):
        # For tree-based models
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(f"\nTop 10 most important features ({best_model_name}):")
        print(feature_importance.head(10))
    elif best_model_name == "Logistic Regression":
        # For logistic regression
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Coefficient': best_model.coef_[0]
        }).sort_values('Coefficient', ascending=False)
        print(f"\nTop 10 most influential features ({best_model_name}):")
        print(feature_importance.head(10))
    
    return models

# Train and evaluate models
models = train_and_evaluate_models(X_train, X_test, y_train, y_test, features)

def visualize_model_performance(models, X_test, y_test):
    """
    Create advanced visualizations for model performance comparison
    
    Parameters:
    -----------
    models : dict
        Dictionary containing models and their evaluation metrics
    X_test : numpy.ndarray
        Testing features
    y_test : pandas.Series
        Testing target values
    """
    print("="*50)
    print("MODEL PERFORMANCE VISUALIZATION")
    print("="*50)
    
    # 1. ROC Curves for all models
    plt.figure(figsize=(12, 8))
    for name, model_dict in models.items():
        y_proba = model_dict["probabilities"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = model_dict["metrics"]["roc_auc"]
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for All Models', fontsize=15)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 2. Precision-Recall Curves
    plt.figure(figsize=(12, 8))
    for name, model_dict in models.items():
        y_proba = model_dict["probabilities"]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, linewidth=2, label=f'{name}')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves for All Models', fontsize=15)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 3. Confusion Matrices for the best model
    best_model_name = max(models, key=lambda x: models[x]["metrics"]["roc_auc"])
    y_pred = models[best_model_name]["predictions"]
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Diabetes', 'Diabetes'])
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=15)
    plt.show()
    
    # 4. Model metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    model_names = list(models.keys())
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        values = [models[name]["metrics"][metric] for name in model_names]
        bars = plt.bar(model_names, values, color=sns.color_palette('viridis', len(model_names)))
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=11)
        
        plt.ylim(0, max(values) * 1.1)  # Add some space for the text
        plt.title(f'Model Comparison: {metric.replace("_", " ").title()}', fontsize=15)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.show()
    
    # 5. Feature importance visualization (for the best model)
    best_model = models[best_model_name]["model"]
    
    if hasattr(best_model, 'feature_importances_'):
        # For tree-based models
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        plt.title(f'Feature Importance ({best_model_name})', fontsize=15)
        plt.xlabel('Importance', fontsize=12)
        plt.tight_layout()
        plt.show()
        
    elif best_model_name == "Logistic Regression":
        # For logistic regression
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Coefficient': np.abs(best_model.coef_[0])  # Absolute value for visualization
        }).sort_values('Coefficient', ascending=False).head(15)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Coefficient', y='Feature', data=feature_importance, palette='viridis')
        plt.title(f'Absolute Coefficient Magnitude ({best_model_name})', fontsize=15)
        plt.xlabel('|Coefficient|', fontsize=12)
        plt.tight_layout()
        plt.show()

# Visualize model performance
visualize_model_performance(models, X_test, y_test)

def save_and_demonstrate_model(models, scaler, features):
    """
    Save the best model and demonstrate how to use it for predictions
    
    Parameters:
    -----------
    models : dict
        Dictionary containing models and their evaluation metrics
    scaler : sklearn.preprocessing.PowerTransformer
        Fitted scaler for feature preprocessing
    features : list
        List of feature names
    """
    print("="*50)
    print("MODEL SAVING AND DEPLOYMENT DEMONSTRATION")
    print("="*50)
    
    # Identify the best model based on ROC AUC
    best_model_name = max(models, key=lambda x: models[x]["metrics"]["roc_auc"])
    best_model = models[best_model_name]["model"]
    
    print(f"\nBest model: {best_model_name}")
    
    # Save the model and scaler
    model_filename = 'diabetes_prediction_model.pkl'
    scaler_filename = 'diabetes_prediction_scaler.pkl'
    
    joblib.dump(best_model, model_filename)
    joblib.dump(scaler, scaler_filename)
    
    # Save feature list
    with open('feature_list.txt', 'w') as f:
        f.write('\n'.join(features))
    
    print(f"Model saved to {model_filename}")
    print(f"Scaler saved to {scaler_filename}")
    print(f"Feature list saved to feature_list.txt")
    
    # Demonstrate how to load and use the model
    print("\nDemonstration of model loading and prediction:")
    
    # Load the model and scaler
    loaded_model = joblib.load(model_filename)
    loaded_scaler = joblib.load(scaler_filename)
    
    # Example patient data (in the original feature space, before feature engineering)
    example_data = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    
    print("\nExample patient data:")
    for k, v in example_data.items():
        print(f"  {k}: {v}")
    
    # In a real application, you would need to:
    # 1. Perform the same feature engineering as during training
    # 2. Transform the data using the saved scaler
    # 3. Make sure all engineered features are present
    
    print("\nNote: In a production environment, you would need to recreate")
    print("all engineered features and transform using the saved scaler.")
    
    # For demonstration, we'll use a simplified approach:
    # Create a function that would be used in production
    def predict_diabetes_risk(patient_data, model, scaler, original_features):
        """
        Make diabetes prediction for a new patient
        
        Parameters:
        -----------
        patient_data : dict
            Dictionary with patient's medical data
        model : sklearn estimator
            Trained machine learning model
        scaler : sklearn transformer
            Fitted scaler
        original_features : list
            List of original feature names
            
        Returns:
        --------
        tuple
            (prediction, probability)
        """
        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_data])
        
        # Handle missing values (zeros)
        medical_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for feature in medical_features:
            if feature in patient_df.columns and patient_df[feature].iloc[0] == 0:
                patient_df[feature] = np.nan
        
        # Use simple imputation for this example (in production would use the same imputer)
        for feature in medical_features:
            if feature in patient_df.columns and pd.isna(patient_df[feature]).any():
                # Use median from training data (this would be stored in production)
                patient_df[feature].fillna(df[feature].median(), inplace=True)
        
        # Feature engineering (simplified version)
        # BMI categories
        patient_df['BMI_Category_Normal'] = 0
        patient_df['BMI_Category_Overweight'] = 0
        patient_df['BMI_Category_Obese'] = 0
        
        bmi = patient_df['BMI'].iloc[0]
        if 18.5 <= bmi <= 24.9:
            patient_df['BMI_Category_Normal'] = 1
        elif 25 <= bmi <= 29.9:
            patient_df['BMI_Category_Overweight'] = 1
        elif bmi >= 30:
            patient_df['BMI_Category_Obese'] = 1
        
        # Glucose-Insulin interaction
        patient_df['Glucose_Insulin_Ratio'] = patient_df['Glucose'] / (patient_df['Insulin'] + 1)
        
        # Age groups
        patient_df['Age_Group_30-40'] = 0
        patient_df['Age_Group_40-50'] = 0
        patient_df['Age_Group_50+'] = 0
        
        age = patient_df['Age'].iloc[0]
        if 30 <= age < 40:
            patient_df['Age_Group_30-40'] = 1
        elif 40 <= age < 50:
            patient_df['Age_Group_40-50'] = 1
        elif age >= 50:
            patient_df['Age_Group_50+'] = 1
        
        # Make sure all required features are present
        for feature in features:
            if feature not in patient_df.columns:
                patient_df[feature] = 0
        
        # Keep only the features used by the model
        patient_df = patient_df[features]
        
        # Scale the features
        scaled_features = scaler.transform(patient_df)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]
        
        return prediction, probability
    
    # Make prediction
    prediction, probability = predict_diabetes_risk(
        example_data, loaded_model, loaded_scaler, df.columns.tolist()
    )
    
    print(f"\nPrediction: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
    print(f"Probability of diabetes: {probability:.4f} ({probability:.2%})")
    
    # Risk interpretation
    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.7:
        risk_level = "Moderate"
    else:
        risk_level = "High"
    
    print(f"Risk level: {risk_level}")
    
    print("\nIn a production environment, this function would be integrated")
    print("into an API or application to provide real-time predictions.")

# Save model and demonstrate prediction
save_and_demonstrate_model(models, scaler, features)

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Load and explore data
    df = load_and_explore_data("diabetes.csv")
    
    # 2. Visualize data
    visualize_data(df)
    
    # 3. Preprocess data
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(df)
    
    # 4. Train and evaluate models
    models = train_and_evaluate_models(X_train, X_test, y_train, y_test, features)
    
    # 5. Visualize model performance
    visualize_model_performance(models, X_test, y_test)
    
    # 6. Save and demonstrate model
    save_and_demonstrate_model(models, scaler, features)

if __name__ == "__main__":
    main()


