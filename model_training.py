import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load and encode data
df = pd.read_csv('Stress_Dataset.csv')
def encode_categoricals(df):
    df_encoded = df.copy()
    if 'Gender' in df_encoded.columns:
        df_encoded['Gender'] = df_encoded['Gender'].astype('category').cat.codes
    if 'Which type of stress do you primarily experience?' in df_encoded.columns:
        df_encoded['StressType'] = df_encoded['Which type of stress do you primarily experience?'].astype('category').cat.codes
    return df_encoded

df_encoded = encode_categoricals(df)

# Features and target
X = df_encoded.drop(['Which type of stress do you primarily experience?', 'StressType'], axis=1, errors='ignore')
y = df_encoded['StressType']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(clf, 'stress_model.joblib')
print('Model saved as stress_model.joblib')
