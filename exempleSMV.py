import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

# Define a mapping from letters to integer codes
le = LabelEncoder()
le.fit(list(map(chr, range(ord('A'), ord('Z')+1))))

# Define a function to encode a word as a vector
def word_to_vector(word):
    vec = np.zeros(26 * len(word))
    for i, char in enumerate(word):
        vec[i * 26 + le.transform([char])[0]] = 1
    return vec

print(word_to_vector('ABC'))

# Encode a list of words as a matrix of vectors
words = ['AAA', 'ABC', 'ZZZ']
X = np.array([word_to_vector(word) for word in words])
print(X)

# Train a linear SVM classifier on the encoded data
clf = svm.SVC(kernel='linear')
clf.fit(X, [0, 1, 0])

# Use the trained classifier to predict the class of a new word
print(clf.predict(word_to_vector('ABB')))