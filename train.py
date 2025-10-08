""" This file is used to train the model and save it as model.h5 file. """

# Importing the required libraries
import nltk
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer
lemma = WordNetLemmatizer()

# Preprocessing the data
words = []
classes = []
docs = []
ignore_words = ['?', '!', '', "'"]
data_file = open('intents.json').read()
intents = json.loads(data_file)

# Tokenizing the words and building the dataset
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        docs.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatizing and cleaning words
words = [lemma.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))

# Sorting the classes
classes = sorted(set(classes))

# Printing statistics
print(f"{len(docs)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words: {words}")

# Saving the words and classes in pickle files
with open('word.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('class.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Creating the training data
training = []
output_empty = [0] * len(classes)

# Creating the bag of words
for doc in docs:
    bag = [1 if w in doc[0] else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffling the training data
random.shuffle(training)
training = np.array(training, dtype=object)

x_train = np.array(list(training[:, 0]))
y_train = np.array(list(training[:, 1]))

print("Created training data successfully")

# Creating the model: Sequential model
model = Sequential()
model.add(Dense(150, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Compiling the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fitting the model
model.fit(x_train, y_train, epochs=250, batch_size=5, verbose=1)

# Saving the model
model.save('model.h5')

print("Successful model creation")

# Evaluating the model
loss, accuracy = model.evaluate(x_train, y_train)
print(f'Accuracy: {accuracy}')
print(f'Loss: {loss}')
