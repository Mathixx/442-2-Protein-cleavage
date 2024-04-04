###########################
### MODULES NECESSAIRES ###
###########################


# Import the necessary libraries
import numpy as np
import pandas as pd
import math

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

from auxFonctions import AminoAcid

# Read data from a file into a list of entries
with open('data/EUKSIG_13.red', 'r') as file:
    entries = file.read().split('\n   ')


# Define a function to process each entry in the data file
def process_entry(entry):
    lines = entry.split('\n')
    protein_id, primary_structure, annotation = lines
    return {
        'Protein ID': protein_id.split()[1],
        'Primary Structure': primary_structure,
        'Annotation': annotation
    }

##############################
## RECUPERATION DES DONNÉES ##
##############################


# Process each entry
processed_entries = [process_entry(entry) for entry in entries]

# Create a DataFrame
df = pd.DataFrame(processed_entries)

# Get the position of the cleavage site
cleavage_site_position = df['Annotation'].apply(lambda x: x.find('C'))

# Split the primary structure into a list of amino acids
amino_acid_seq = df['Primary Structure'].apply(lambda x: list(x))

# for each amino acid in the sequence, replace it with the corresponding AminoAcid object
#amino_acid_seq = amino_acid_seq.apply(lambda x: [AminoAcid(aa) for aa in x])

#############################
## TRAITEEMENT DES DONNÉES ##
#############################

# Define a mapping from letters to integer codes
le = LabelEncoder()
le.fit(list(map(chr, range(ord('A'), ord('Z')+1))))

# Define a function to encode a word as a vector
def word_to_vector(word):
    vec = np.zeros(26 * len(word))
    for i, char in enumerate(word):
        vec[i * 26 + le.transform([char])[0]] = 1
    return vec

"""
A FAIRE :
ON FIXE P ET Q, ON PEUT PRENDRE COMME DANS PARTIE 1

POUR CHAQUE SEQUENCE ON EXTRAIT LE MOT DE TAILLE P+Q AUTOUR DE LA CLEAVAGE SITE
MAIS AUSSI LES MOTS qui ne sont PAS AUTOUR DE LA CLEAVAGE SITE

ON construit un vecteur de taille 26*(P+Q) pour chaque mot
ON recupere AINSI UN ENSEMBLE DE DONNÉES X

ON RECUPERE AUSSI LES LABELS Y =? 0 OU 1 en fonction de la position de la cleavage site
Et inchallah on fait un SVM
"""



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(amino_acid_seq, cleavage_site_position, test_size=0.2, random_state=42)
"""
test_size=0.2: This argument specifies the proportion of the dataset to include in the test split. 
In this case, 20% of the data will be used for testing, and the remaining 80% will be used for training

random_state=42: This argument sets the seed for the random number generator that shuffles the data before splitting. 
Setting a specific seed (like 42 in this case) ensures that the output is reproducible, i.e., 
you'll get the same train/test split each time you run the code.
"""

print(y_train)
# Define the SVM classifier
classifier = svm.SVC(kernel='rbf')

#edouar suce mq bite


"""
# Fit the classifier to the training data
classifier.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred = classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
"""






# Encode the amino acid sequences as vectors
encoded_sequences = amino_acid_seq.apply(lambda seq: [1 if aa == AminoAcid('A' + i) else 0 for aa in seq for i in range(26)])

# Fit the classifier to the training data
classifier.fit(encoded_sequences[X_train.index], y_train)

# Encode the test sequences
encoded_test_sequences = X_test.apply(lambda seq: [1 if aa == AminoAcid('A' + i) else 0 for aa in seq for i in range(26)])

# Predict the target variable for the encoded test sequences
y_pred = classifier.predict(encoded_test_sequences)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)