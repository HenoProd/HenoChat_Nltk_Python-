import nltk
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Charger les données à partir du fichier JSON
with open('intens.json', 'r') as file:
    intents = json.load(file)

# Prétraitement des données
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

patterns = []
responses = []
tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern.lower())
        responses.append(intent['responses'])
        tags.append(intent['tag'])

# Entraîner le classificateur
vectorizer = CountVectorizer(preprocessor=preprocess_text)
X = vectorizer.fit_transform(patterns)
y = tags

classifier = MultinomialNB()
classifier.fit(X, y)

# Fonction pour prédire la réponse
def get_response(user_input):
    user_input = preprocess_text(user_input)
    user_vector = vectorizer.transform([user_input])
    predicted_tag = classifier.predict(user_vector)[0]

    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return intent['responses'][0]

# Interaction avec l'utilisateur
while True:
    user_input = input("Vous: ")
    if user_input.lower() == 'exit':
        break
    print("Chatbot:", get_response(user_input))
