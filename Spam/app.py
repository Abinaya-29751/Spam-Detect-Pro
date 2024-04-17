import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flask import Flask, render_template, request

# Load the dataset
df = pd.read_csv(r"C:\Users\HP\Desktop\INTERNSHIP\PROJECTS\Spam\data\spam.csv", encoding='latin1')

# Drop unnecessary columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# Rename columns
df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)

# Convert labels to numerical values
df['label'] = df.label.map({'ham': 0, 'spam': 1})

# Data preprocessing and NLP setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Lowercase
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Remove stopwords and perform stemming
    return ' '.join(words)

# Apply preprocessing to text column
df['text'] = df['text'].apply(preprocess_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], random_state=1)

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and transform it
training_data = count_vector.fit_transform(X_train)

# Transform testing data
testing_data = count_vector.transform(X_test)

# Initialize and train the Naive Bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

# Initialize Flask app
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for predicting
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        preprocessed_message = preprocess_text(message)
        input_data = count_vector.transform([preprocessed_message])
        prediction = naive_bayes.predict(input_data)
        if prediction[0] == 0:
            result = 'Not spam'
        else:
            result = 'Spam'
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
