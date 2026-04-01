import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve,accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

# --- Configuration ---
DATA_FILE = "vault_hids_dataset.csv"
RANDOM_STATE = 42 # for reproducibility

# --- Load the dataset ---
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset '{DATA_FILE}' loaded successfully. Shape: {df.shape}")
    print(df.head())
except FileNotFoundError:
    print(f"Error: The file '{DATA_FILE}' was not found. Please run the data generation script first.")
    exit()

# --- Feature Engineering ---
print("\nPerforming Feature Engineering...")

# Convert timestamp to datetime objects
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Calculate time-based features
df['hour_of_day'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek # Monday=0, Sunday=6
df['day_of_year'] = df['timestamp'].dt.dayofyear
df['month'] = df['timestamp'].dt.month
df['minute'] = df['timestamp'].dt.minute
df['second'] = df['timestamp'].dt.second
df['time_in_seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()


# Is it a weekend? (Saturday=5, Sunday=6)
df['is_weekend'] = ((df['timestamp'].dt.dayofweek == 5) | (df['timestamp'].dt.dayofweek == 6)).astype(int)

# --- RFID Features ---
# Define known authorized and unauthorized RFID tags (must match data generation script)
authorized_rfids = ["RFID_VAULT_ADMIN_001", "RFID_SECURITY_002", "RFID_MAINTENANCE_003"]
unauthorized_rfids = ["RFID_UNKNOWN_A", "RFID_UNKNOWN_B", "RFID_BLACKHAT_001"]

# Create a feature for unauthorized RFID detection
df['unauthorized_rfid_detected'] = df['rfid_tag_id'].apply(lambda x: 1 if x in unauthorized_rfids else 0)

# Handle 'NONE' RFID case. 'NONE' could be normal (no one around) or an anomaly (jamming)
# The model will learn from context. For now, treat it as another category.

# Convert rfid_tag_id to categorical using one-hot encoding
# Be careful with `NONE` as a category here
df = pd.get_dummies(df, columns=['rfid_tag_id'], prefix='rfid', dummy_na=False)


# --- Define Features (X) and Target (y) ---
# Drop original timestamp and anomaly_type (for now, as it's for detailed analysis, not direct training label)
# Also drop any highly correlated features if necessary after inspection
features_to_drop = ['timestamp', 'anomaly_type']
X = df.drop(columns=['is_anomaly'] + features_to_drop, errors='ignore')
y = df['is_anomaly']

print("\nFeatures after engineering and one-hot encoding:")
print(X.head())
print(f"Total features: {X.shape[1]}")

# --- Chronological Train-Test Split ---
# This is crucial for time-series data to avoid data leakage
# Train on earlier data, test on later data
print("\nSplitting data chronologically into training and testing sets...")
split_ratio = 0.8 # 80% for training, 20% for testing

# Calculate the split point based on the sorted timestamps
split_index = int(len(df) * split_ratio)
X_train_df = X.iloc[:split_index]
X_test_df = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Align columns - crucial after one-hot encoding if test set misses a category from train
# Or if you used train_test_split without specific order.
# The `reindex` ensures both train and test sets have the same columns, filling missing with 0
# (This step is more critical if using `train_test_split` randomly. With chronological split,
# assuming data covers full range, it might be less of an issue, but good practice.)
train_cols = X_train_df.columns
X_test_df = X_test_df.reindex(columns=train_cols, fill_value=0)

X_train = X_train_df.values
X_test = X_test_df.values


print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"Anomaly distribution in training set:\n{pd.Series(y_train).value_counts(normalize=True)}")
print(f"Anomaly distribution in test set:\n{pd.Series(y_test).value_counts(normalize=True)}")


# --- Train the Random Forest Model ---
print("\nTraining Random Forest Classifier...")
# Using class_weight='balanced' to handle potential imbalance in anomaly class
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_train, y_train)
print("Model training complete.")

# --- Evaluate the Model ---
print("\nEvaluating the model on the test set...")
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1] # Probability of being the positive class (1 = anomaly)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Anomaly (1)']))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Anomaly'],
            yticklabels=['True Normal', 'True Anomaly'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

print(f"\nROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(y_test, y_proba):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# --- Feature Importance ---
print("\n--- Feature Importance ---")
feature_importances = pd.DataFrame({'feature': X_train_df.columns, 'importance': rf_model.feature_importances_})
feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)
print(feature_importances.head(10)) # Print top 10 important features

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances.head(15))
plt.title('Top 15 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("\nModel training and evaluation complete.")
# Save the model
joblib.dump(rf_model, "my_random_forest_ids_model.pkl")
print("Model saved!")

# Load model
model = joblib.load("my_random_forest_ids_model.pkl")

# def detect_anomaly(features):
#     prediction = model.predict(features)
#     return prediction[0]

# # 4. Main Loop: Live IDS
# while True:
#     features = prepare_features()
#     result = detect_anomaly(features)
    
#     if result == 1:
#         print("[ALERT] Possible Intrusion Detected!")
#     else:
#         print("[INFO] Normal Activity")
    
#     time.sleep(1)