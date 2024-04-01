
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
from aux import AminoAcid

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
#print("Position of the cleavage site:")
#print(cleavage_site_position)
print("Average position of the cleavage site:")
print(cleavage_site_position.mean())
print("\n")


"""
Over a set of N sequences, with a known cleavage site for each, one can first count c(a, i), 
the number of occurrences of each amino acid a ∈ A, at every position i ∈ {−p, ..., q − 1}, 
relative to the corresponding cleavage site. 
We are facing a binary classification problem. 
Given any whole protein sequence (ai)i=0,...,l−1, and any position j, where p ≤ j ≤ l−q, 
the word aj−paj−p+1 · · · aj−1aj · · · aj+q−1 ∈ Ap+q should be enough to decide 
whether the bond at position j, between aj−1 and aj, is a cleavage site or not.

Then, for each a and i, let define f(a,i) = c(a,i)/N, 
the observed frequency of amino acid a at the relative position i.
"""

# Split the primary structure into a list of amino acids
amino_acid_seq = df['Primary Structure'].apply(lambda x: list(x))

# for each amino acid in the sequence, replace it with the corresponding AminoAcid object
amino_acid_seq = amino_acid_seq.apply(lambda x: [AminoAcid(aa) for aa in x])


p = 13
q = 2

# Create a DataFrame to store the counts of each amino acid at every position relative to the cleavage site
#the cleavage site is between to aminoacids, so cleavage_site_position is the position of the first amino acid after the cleavge site
#So i need to create a dataframe with columns from -p to q without 0
amino_acid_counts = pd.DataFrame(0, index=AminoAcid.properties.keys(), columns=range(-p, q))
amino_acid_freqs = pd.DataFrame(0, index=AminoAcid.properties.keys(), columns=range(-p, q))
amino_acid_pseudo_counts = pd.DataFrame(0, index=AminoAcid.properties.keys(), columns=range(-p, q))
amino_acid_s_values = pd.DataFrame(0, index=AminoAcid.properties.keys(), columns=range(-p, q))


# Count the occurrences of each amino acid at every position relative to the cleavage site

for i, seq in amino_acid_seq.items():
    for j, aa in enumerate(seq):
        position = j - cleavage_site_position[i] #position of the amino acid relative to the cleavage site
        if position in amino_acid_counts.columns:
            amino_acid_counts.loc[aa.code, position] += 1

# Add pseudo-counts to avoid zero counts here pseudocount parameter is 1/len(df)
amino_acid_pseudo_counts = amino_acid_counts + 1

# Print the results
print("Occurrences of each amino acid at every position relative to the cleavage site:")
print(amino_acid_pseudo_counts)

# Compute the observed frequency of each amino acid at the relative position (using pseudo-counts)
for i in amino_acid_counts.index:
    for j in amino_acid_counts.columns:
        amino_acid_freqs.loc[i, j] = amino_acid_pseudo_counts.loc[i, j] / len(df)

# Compute the general background frequency of each amino acid
general_background_frequency = amino_acid_freqs.mean(axis=1)

# Compute the s value of each amino acid at every position
for i in amino_acid_counts.index:
    for j in amino_acid_counts.columns:
        amino_acid_s_values.loc[i, j] = math.log(amino_acid_freqs.loc[i, j]) - math.log(general_background_frequency[i])

#Finally, for any word w = a0a1 · · · ap+q−1,
#the q − 1 score defined as Pq−1 s(ap+i, i) may tell whether w is the neighborhood of a cleavage i=−p
#site or not.
        
# Define the function computing the q-1 score for a given word
def q_minus_1_score(word):
    return sum([amino_acid_s_values.loc[aa.code, i-13] for i, aa in enumerate(word)])

threshold = 0

#A simple thresholding (to be tuned) is then enough to define a simple binary classifier.
def is_cleavage_neighborhood(score):
    return score > threshold


#Lets test the function
word = 'A'*(p+q)
print(len(word))
print(q_minus_1_score([AminoAcid(aa) for aa in word]))



#Here I'm saving my results in the result.txt file
def save_results():
    with open('results.txt', 'a') as file:
        file.write("###     RESULTS :#####\n\n")
        file.write("Parameters of the model:\n")
        file.write("Data obtained from the EUKSIG_13.red file\n\n")
        file.write("p = 13    q = 2\n\n")
        file.write("Average position of the cleavage site:\n")
        file.write(str(cleavage_site_position.mean()) + "\n\n")
        file.write("Occurrences of each amino acid at every position relative to the cleavage site:\n")
        amino_acid_counts_str = amino_acid_counts.to_string()
        file.write(amino_acid_counts_str + "\n\n")
        file.write("Observed frequency of each amino acid at the relative position:\n")
        amino_acid_freqs_str = amino_acid_freqs.to_string()
        file.write(amino_acid_freqs_str + "\n\n")
        file.write("General background frequency of each amino acid:\n")
        file.write(str(general_background_frequency) + "\n\n")
        file.write("s value of each amino acid at every position:\n")
        amino_acid_s_values_str = amino_acid_s_values.to_string()
        file.write(amino_acid_s_values_str + "\n\n")
        file.write("END OF RESULTS\n\n\n")
    
#save_results()


