"""
Over a set of N sequences, with a known cleavage site for each, one can first count c(a, i), the number of 
occurrences of each amino acid a ∈ A, at every position i ∈ {−p, ..., q − 1}, relative to the corresponding 
cleavage site. Then, for each a and i, let define f(a,i) = c(a,i)/N, the observed frequency of amino acid a at the 
relative position i.

In a same way, by counting over the whole length of given sequences, one can compute the observed general background 
frequency g(a) of amino acid a in the given set, regardless of the position. However, it must be noticed that the very 
first amino acid at the beginning of a sequence is almost always an M, because it corresponds to the transcription of 
the start codon. Also, one will not count letters on this first position to avoid a bias.

These frequencies will be used as estimated probabilities to compute the probability of a given word to be 
located at a cleavage site, under an independent model. We rather use the logarithm of probabilities to go on 
additive calculations.

Then, ∀a ∈ A,∀i ∈ {-p,...,q-1}, we define s(a,i) = log(f(a,i)) - log(g(a)). Also, as zero
counts may occur, pseudo-counts [3] must be used. Finally, for any word w = a0a1 · · · a(p+q−1),
the q − 1 score defined as sum for i in [-p, ..,q-1] of s(a(p+i), i) may tell whether w is the neighborhood of a 
cleavage i=−p
site or not. A simple thresholding (to be tuned) is then enough to define a simple binary classifier.

"""

import pandas as pd
import math
from aux import *

# Read data from a file into a list of entries
with open('/Users/mathiasperez/Documents/GitHub/442-2-Protein-cleavage/data/EUKSIG_13.red', 'r') as file:
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


# Process each entry
processed_entries = [process_entry(entry) for entry in entries]

# Create a DataFrame
df = pd.DataFrame(processed_entries)

# Now you can analyze the DataFrame as needed
# For example, you could count the occurrences of each amino acid in the primary structure
"""
amino_acid_counts = df['Primary Structure'].apply(lambda x: pd.Series(list(x))).stack().value_counts()
"""


# Get the position of the cleavage site
cleavage_site_position = df['Annotation'].apply(lambda x: x.find('C'))
print("Position of the cleavage site:")
print(cleavage_site_position)


"""
Over a set of N sequences, with a known cleavage site for each, one can first count c(a, i), 
the number of occurrences of each amino acid a ∈ A, at every position i ∈ {−p, ..., q − 1}, 
relative to the corresponding cleavage site. 
Then, for each a and i, let define f(a,i) = c(a,i)/N, 
the observed frequency of amino acid a at the relative position i.
"""

# Count the occurrences of each amino acid at every position relative to the cleavage site
amino_acid_counts = df['Primary Structure'].apply(lambda x: pd.Series(list(x))).stack().reset_index(drop=True).to_frame('Amino Acid')

print(amino_acid_counts)

amino_acid_counts['Position'] = amino_acid_counts.groupby(level=0).cumcount() - cleavage_site_position
amino_acid_counts = amino_acid_counts.groupby(['Amino Acid', 'Position']).size().unstack(fill_value=0)

# Calculate the observed frequency of each amino acid at the relative position
observed_frequency = amino_acid_counts.div(len(df))

# Calculate the logarithm of the observed frequency
log_observed_frequency = observed_frequency.applymap(lambda x: math.log(x) if x > 0 else float('-inf'))

# Compute the general background frequency of each amino acid
general_background_frequency = observed_frequency.mean()

# Calculate the logarithm of the general background frequency
log_general_background_frequency = general_background_frequency.apply(lambda x: math.log(x) if x > 0 else float('-inf'))

# Calculate the score for each amino acid at every position
score = log_observed_frequency.subtract(log_general_background_frequency, axis=1)

# Calculate the q-1 score for each word
q_minus_1_score = score.sum(axis=1)

# Define a threshold for the binary classifier
threshold = 0

# Classify each word as cleavage site or not based on the q-1 score
df['Cleavage Site'] = q_minus_1_score > threshold

# Print the results
print("Position of the cleavage site:")

"""
print("Counts of each amino acid:")
print(amino_acid_counts)
print("\nObserved frequency of each amino acid at the relative position:")
print(observed_frequency)
print("\nLogarithm of the observed frequency:")
print(log_observed_frequency)
print("\nGeneral background frequency of each amino acid:")
print(general_background_frequency)
print("\nLogarithm of the general background frequency:")
print(log_general_background_frequency)
print("\nScore for each amino acid at every position:")
print(score)
print("\nq-1 score for each word:")
print(q_minus_1_score)
print("\nClassification result:")
print(df[['Primary Structure', 'Cleavage Site']])
"""


"""
AminoList = { }
# Apply the function to each row in the DataFrame
test = df['Distance from Cleavage Site'] = df.apply(lambda row: distance_from_cleavage_site(row['Primary Structure'], row['Annotation']), axis=1)

# Now you can analyze the DataFrame to see which amino acid is closer to the cleavage site
print("distance from cleavage site :\n")
print(test)
"""


