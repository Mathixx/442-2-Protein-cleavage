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

# print(amino_acid_seq)
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





def convert_df_to_vectors(df):
    '''
    Convert the dataframe to a format that can be used for training a classifier
    '''
    df_exploitable = df.copy()
    df_exploitable['Annotation_vector'] = df_exploitable.apply(lambda x: [1 if i==x['Annotation'].find('C') else 0 for i in range(len(x['Annotation']))], axis=1)
    df_exploitable['P_Structure_vector'] = df_exploitable['Primary Structure'].apply(word_to_vector)
    return df_exploitable


df_exploitable = convert_df_to_vectors(df)


# Extract a random subsequence of length n

n = 12  # length of the subsequence
nb_letters = 26  # number of different letters in the alphabet
def extract_random_subsequence(row, n):
    max_start_index = max(0, len(row['Primary Structure']) - n)  # Calculate the maximum possible start index
    if max_start_index == 0:
        start_index = 0  # if chain is too short, start at the beginning
    else:
        start_index = np.random.randint(0, max_start_index)  # Randomly select a start index
    end_index = start_index + n  # Calculer l'indice de fin
    return pd.Series([row['Primary Structure'][start_index:end_index], row['Annotation'][start_index:end_index], row['P_Structure_vector'][start_index*26:end_index*26], row['Annotation_vector'][start_index:end_index]], index=['Primary Structure', 'Annotation', 'P_Structure_vector', 'Annotation_vector'])

# Extract random subsequences of length n from the data
df_random_subsequence = df_exploitable.apply(extract_random_subsequence, axis=1, n=n)
print(type(df_random_subsequence['P_Structure_vector']))

# Split the random subsequence data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_random_subsequence['P_Structure_vector'], df_random_subsequence['Annotation_vector'], test_size=0.2, random_state=42)
"""
test_size=0.2: This argument specifies the proportion of the dataset to include in the test split. 
In this case, 20% of the data will be used for testing, and the remaining 80% will be used for training

random_state=42: This argument sets the seed for the random number generator that shuffles the data before splitting. 
Setting a specific seed (like 42 in this case) ensures that the output is reproducible, i.e., 
you'll get the same train/test split each time you run the code.
"""


classifier = svm.SVC(kernel='rbf')
classifier.fit(X_train, y_train)


# print("Accuracy:", accuracy)



# print(y_train)
# # Define the SVM classifier
# classifier = svm.SVC(kernel='rbf')




"""
# Fit the classifier to the training data
classifier.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred = classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
"""

# # Encode the amino acid sequences as vectors
# acides_amines = 'ACDEFGHIKLMNPQRSTVWY*'
# mapping = {aa: np.eye(len(acides_amines))[i] for i, aa in enumerate(acides_amines)}

# def encoder_sequence(sequence):
#     # Encodage de la séquence en utilisant le mapping
#     return np.array([mapping[aa] for aa in sequence if aa in mapping])

# encoded_sequences = amino_acid_seq.apply(encoder_sequence)

# # Fit the classifier to the training data
# # classifier.fit(X_train, y_train)


