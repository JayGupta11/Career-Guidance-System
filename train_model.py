import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

try:
    df = pd.read_csv('cs_students.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'cs_students.csv' not found. Please ensure it is in the folder.")
    exit()

df = df.drop(['Student ID', 'Name', 'Major', 'Projects'], axis=1)

print(f"Original Data Size: {len(df)} rows")
print("Augmenting data to simulate larger population...")

df = pd.concat([df] * 30, ignore_index=True)

df['Age'] = df['Age'] + np.random.randint(-1, 2, size=len(df))
df['GPA'] = df['GPA'] + np.random.normal(0, 0.05, size=len(df))
df['GPA'] = df['GPA'].clip(0.0, 4.0)

print(f"Augmented Data Size: {len(df)} rows")

encoders = {}

skill_map = {'Weak': 1, 'Average': 2, 'Strong': 3}
for col in ['Python', 'SQL', 'Java']:
    df[col] = df[col].map(skill_map)

for col in ['Gender', 'Interested Domain', 'Future Career']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop('Future Career', axis=1)
y = df['Future Career']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Model on Augmented Data...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print("-" * 30)
print(f"RESEARCH RESULTS:")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")
print("-" * 30)

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metrics_values = [accuracy, precision, recall, f1]

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics_names, metrics_values, color=['#4F46E5', '#10B981', '#F59E0B', '#EF4444'])
plt.ylim(0, 1.1)
plt.title('Augmented Model Performance Metrics', fontsize=16)
plt.ylabel('Score')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=12)
plt.savefig('model_metrics_chart.png')
print("Saved: model_metrics_chart.png")
plt.close()

plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
target_names = encoders['Future Career'].classes_
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix (Career Prediction)', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved: confusion_matrix.png")
plt.close()

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("Model saved successfully! Now run 'python app.py'")