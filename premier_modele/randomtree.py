import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

train_df = pd.read_csv("train_dataset_final.csv")
val_df = pd.read_csv("val_dataset_final.csv")

feature_cols = ['dbh_approx','p10_height_rel', 'p50_height_rel', 'p90_height_rel','height','num_points', 'crown_diameter', 'point_density', 'volume', 'crown_area']
target_col = 'species'

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_val = val_df[feature_cols]
y_val = val_df[target_col]

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train_encoded)

print("Entrainement complet")

y_pred = rf_model.predict(X_val)
accuracy = accuracy_score(y_val_encoded, y_pred)

print(f"\nModel Accuracy: {accuracy:.2%}")
print("-" * 30)
print("Classification Report:")
print(classification_report(y_val_encoded, y_pred, target_names=le.classes_))

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance_df)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_val_encoded, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Species')
plt.ylabel('Actual Species')
plt.title('Confusion Matrix (Validation Set)')
plt.show()