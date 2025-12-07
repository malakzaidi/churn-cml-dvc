## main
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from imblearn.over_sampling import SMOTE
import joblib
from PIL import Image

## sklearn -- preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector

## sklearn -- models
from sklearn.ensemble import RandomForestClassifier

## sklearn -- metrics
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

## --------------------- Data Preparation ---------------------------- ##

# Read dataset
TRAIN_PATH = os.path.join(os.getcwd(), 'data/dataset.csv')
df = pd.read_csv(TRAIN_PATH)

# Drop first 3 features
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Filter by age
df.drop(index=df[df['Age'] > 80].index.tolist(), axis=0, inplace=True)

# Split into features and target
X = df.drop(columns=['Exited'], axis=1)
y = df['Exited']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=45, stratify=y
)

## --------------------- Data Processing ---------------------------- ##

num_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
categ_cols = ['Gender', 'Geography']
ready_cols = list(set(X_train.columns.tolist()) - set(num_cols) - set(categ_cols))

# Pipelines
num_pipeline = Pipeline(steps=[
    ('selector', DataFrameSelector(num_cols)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categ_pipeline = Pipeline(steps=[
    ('selector', DataFrameSelector(categ_cols)),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(drop='first', sparse_output=False))
])

ready_pipeline = Pipeline(steps=[
    ('selector', DataFrameSelector(ready_cols)),
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Combine
all_pipeline = FeatureUnion(transformer_list=[
    ('numerical', num_pipeline),
    ('categorical', categ_pipeline),
    ('ready', ready_pipeline)
])

# Transform
X_train_final = all_pipeline.fit_transform(X_train)
X_test_final = all_pipeline.transform(X_test)

## --------------------- Imbalance Handling ---------------------------- ##

# Class weights
vals_count = 1 - (np.bincount(y_train) / len(y_train))
vals_count = vals_count / np.sum(vals_count)
dict_weights = {i: vals_count[i] for i in range(2)}

# SMOTE oversampling
over = SMOTE(sampling_strategy=0.7, random_state=45)
X_train_resampled, y_train_resampled = over.fit_resample(X_train_final, y_train)

## --------------------- Set up Saving Directories ---------------------------- ##

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Clear metrics.txt
with open('metrics.txt', 'w') as f:
    f.write('```\n')

## --------------------- Training Function ---------------------------- ##

def train_model(X_train, y_train, plot_name='', class_weight=None):
    """Train a RandomForest model and save it."""
    global clf_name

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        random_state=45,
        class_weight=class_weight
    )

    clf.fit(X_train, y_train)

    # Predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test_final)

    # Metrics
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)

    precision_train = precision_score(y_train, y_pred_train)
    precision_test = precision_score(y_test, y_pred_test)

    recall_train = recall_score(y_train, y_pred_train)
    recall_test = recall_score(y_test, y_pred_test)

    clf_name = clf.__class__.__name__

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{clf_name}_{plot_name}.pkl")
    joblib.dump(clf, model_path)

    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cbar=False, fmt='.0f', cmap='Blues')
    plt.title(f'{plot_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(ticks=np.arange(2) + 0.5, labels=['No Churn', 'Churn'])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=['No Churn', 'Churn'])
    plt.savefig(f'{plot_name}.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Write metrics
    with open('metrics.txt', 'a') as f:
        f.write(f'\n{"=" * 60}\n')
        f.write(f'Model: {clf_name}\n')
        f.write(f'Configuration: {plot_name}\n')
        f.write(f'Saved to: {model_path}\n')
        f.write(f'{"=" * 60}\n\n')
        f.write(f'Training Set:\n')
        f.write(f'  - F1-Score:  {f1_train * 100:6.2f}%\n')
        f.write(f'  - Precision: {precision_train * 100:6.2f}%\n')
        f.write(f'  - Recall:    {recall_train * 100:6.2f}%\n\n')
        f.write(f'Test Set:\n')
        f.write(f'  - F1-Score:  {f1_test * 100:6.2f}%\n')
        f.write(f'  - Precision: {precision_test * 100:6.2f}%\n')
        f.write(f'  - Recall:    {recall_test * 100:6.2f}%\n')

    print(f"📦 Model saved → {model_path}")
    return True

## --------------------- Train Models ---------------------------- ##

print("Training model without imbalance handling...")
train_model(X_train=X_train_final, y_train=y_train, plot_name='without-imbalance', class_weight=None)

print("Training model with class weights...")
train_model(X_train=X_train_final, y_train=y_train, plot_name='with-class-weights', class_weight=dict_weights)

print("Training model with SMOTE oversampling...")
train_model(X_train=X_train_resampled, y_train=y_train_resampled, plot_name='with-SMOTE', class_weight=None)

# Close code block
with open('metrics.txt', 'a') as f:
    f.write(f'\n{"=" * 60}\n```\n')

## --------------------- Combine Confusion Matrices ---------------------------- ##

confusion_matrix_paths = ['./without-imbalance.png', './with-class-weights.png', './with-SMOTE.png']

plt.figure(figsize=(18, 6))
for i, path in enumerate(confusion_matrix_paths, 1):
    img = Image.open(path)
    plt.subplot(1, len(confusion_matrix_paths), i)
    plt.imshow(img)
    plt.axis('off')

plt.suptitle(f'{clf_name} - Confusion Matrices Comparison', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('conf_matrix.png', dpi=300)
plt.close()

# Delete old images
for path in confusion_matrix_paths:
    if os.path.exists(path):
        os.remove(path)

print("\n✅ Training completed successfully!")
print("📊 Metrics saved to: metrics.txt")
print("📦 Models saved inside /models/")
print("🖼️ Confusion matrices saved to: conf_matrix.png")
