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

# Define a function to process each entry
def process_entry(entry):
    lines = entry.split('\n')
    protein_id, primary_structure, annotation = lines
    return {
        'Protein ID': protein_id.split()[1],
        'Primary Structure': primary_structure,
        'Annotation': annotation
    }

# Read data from a file into a list of entries
with open('/Users/mathiasperez/Documents/GitHub/442-2-Protein-cleavage/data/EUKSIG_13.red', 'r') as file:
    entries = file.read().split('\n   ')


# Process each entry
processed_entries = [process_entry(entry) for entry in entries]

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

# Define a function to calculate the distance of each amino acid from the cleavage site
def distance_from_cleavage_site(primary_structure, annotation,AminoList):
    # Find the position of the cleavage site
    cleavage_site_end = annotation.find('C')
    
    # Calculate the distance  can be negtaive of each amino acid from the cleavage site
    #not the first one cause it is the Met needed for traduction
    
    for i in range(1, len(primary_structure)):
        distance = (i - cleavage_site_end)
        AminoList[primary_structure[i]].append(distance)
    
    return

# Apply the function to each row in the DataFrame
test = df['Distance from Cleavage Site'] = df.apply(lambda row: distance_from_cleavage_site(row['Primary Structure'], row['Annotation']), axis=1)

# Now you can analyze the DataFrame to see which amino acid is closer to the cleavage site
print("distance from cleavage site :\n")
print(test)

