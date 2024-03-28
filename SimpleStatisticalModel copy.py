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
import re
with open('/Users/mathiasperez/Documents/GitHub/442-2-Protein-cleavage/data/EUKSIG_13.red', 'r') as file:
    x = re.findall("^\s*\d+\s+[A-Z0-9_]+\s+\d+\s+[A-Z0-9_]+\s+.+$^\s*[A-Z]+\n^[A-Z]+\n^[A-Z]+$",file.read() )
print(x)


"""
# Create a DataFrame
df = pd.DataFrame(processed_entries)

# Now you can analyze the DataFrame as needed
# For example, you can count the occurrences of each amino acid in the primary structure
amino_acid_counts = df['Primary Structure'].apply(lambda x: pd.Series(list(x))).stack().value_counts()

# Or you can analyze the annotations to count the number of signal peptides and mature proteins
signal_peptides_count = df['Annotation'].apply(lambda x: x.count('S'))
mature_proteins_count = df['Annotation'].apply(lambda x: x.count('M'))

# Print the results
print("Counts of each amino acid:")
print(amino_acid_counts)
print("\nNumber of signal peptides:")
print(signal_peptides_count)
print("\nNumber of mature proteins:")
print(mature_proteins_count)

"""
