# phishing_detector.py
# Detecting Phishing Websites Using Machine Learning
# Author: K. Manichander

# ------------------------------
# 1. Import Required Libraries
# ------------------------------
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ------------------------------
# 2. Load the Dataset
# ------------------------------
print("Loading dataset...")
df = pd.read_csv("phishing_site_urls.csv")

print("Initial Data:")
print(df.head())

# ------------------------------
# 3. Data Cleaning & Label Encoding
# ------------------------------
# Convert labels to numeric
df['Label'] = df['Label'].map({'bad': 1, 'good': 0})
df.dropna(inplace=True)

# ------------------------------
# 4. Feature Extraction Function
# ------------------------------
def extract_features(url):
    try:
        parsed = urlparse(url)
        features = {
            'url_length': len(url),
            'num_dots': url.count('.'),
            'has_at_symbol': 1 if '@' in url else 0,
            'uses_https': 1 if url.startswith('https') else 0,
            'num_hyphens': url.count('-'),
            'num_slashes': url.count('/'),
            'domain_length': len(parsed.netloc),
        }
        return features
    except:
        # If parsing fails, return defaults
        return {'url_length': 0, 'num_dots': 0, 'has_at_symbol': 0,
                'uses_https': 0, 'num_hyphens': 0, 'num_slashes': 0, 'domain_length': 0}

# Apply feature extraction to all URLs
print("\nExtracting features from URLs...")
feature_data = df['URL'].apply(lambda x: pd.Series(extract_features(str(x))))

# Combine features with labels
final_df = pd.concat([feature_data, df['Label']], axis=1)

print("\nFeature columns:")
print(final_df.head())

# ------------------------------
# 5. Train-Test Split
# ------------------------------
X = final_df.drop('Label', axis=1)
y = final_df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------------------
# 6. Train Random Forest Model
# ------------------------------
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# 7. Evaluate Model
# ------------------------------
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# 8. Test with New URLs
# ------------------------------
test_urls = [
    "https://www.paypal.com/account/login",
    "http://freeoffer-update-account-security.com/login",
    "https://cbith.ac.in/",
    "http://verify-user-login-paytm-account.com"
]

print("\nTesting with new URLs:")
for url in test_urls:
    features = pd.DataFrame([extract_features(url)])
    pred = model.predict(features)[0]
    print(f"{url} → {'Phishing' if pred == 1 else 'Legitimate'}")

# ------------------------------
# 9. Save Model for Future Use
# ------------------------------
joblib.dump(model, "phishing_detector_model.pkl")
print("\nModel saved as phishing_detector_model.pkl")

# ------------------------------
# 10. Summary
# ------------------------------
print("\n✅ Phishing Detection Complete!")
print("This model can now be reused to classify new URLs.")
