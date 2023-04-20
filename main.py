# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the password dataset
data = pd.read_csv('password_dataset.csv', delimiter=',')

# Define password strength labels
label_dict = {
    0: 'Weak',
    1: 'Medium',
    2: 'Strong'
}

# Preprocess the password dataset
data = data.drop_duplicates(subset='password', keep='first')
data = data[data['password'].map(len) >= 8]

# Split data into input and output variables
X = data['password']
y = data['strength']

# Transform the input variable into a feature matrix
vectorizer = TfidfVectorizer(analyzer='char')
X = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train a random forest classifier on the training set
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Evaluate the classifier on the testing set
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

# Save the classifier for later use
import pickle
with open('password_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Test the classifier on a sample password
def predict_strength(password):
    X = vectorizer.transform([password])
    y_pred = classifier.predict(X)[0]
    return label_dict[y_pred]

print(predict_strength('password123')) # Expected output: 'Weak'
print(predict_strength('c0mpl3xP@$$w0rd')) # Expected output: 'Strong'

