class AminoAcid:
    # Dictionary to store amino acid properties
    properties = {
        'A': {'name': 'Alanine', 'label': 1},
        'C': {'name': 'Cysteine', 'label': 2},
        'D': {'name': 'Aspartic Acid', 'label': 3},
        'E': {'name': 'Glutamic Acid', 'label': 4},
        'F': {'name': 'Phenylalanine', 'label': 5},
        'G': {'name': 'Glycine', 'label': 6},
        'H': {'name': 'Histidine', 'label': 7},
        'I': {'name': 'Isoleucine', 'label': 8},
        'K': {'name': 'Lysine', 'label': 9},
        'L': {'name': 'Leucine', 'label': 10},
        'M': {'name': 'Methionine', 'label': 11},
        'N': {'name': 'Asparagine', 'label': 12},
        'P': {'name': 'Proline', 'label': 13},
        'Q': {'name': 'Glutamine', 'label': 14},
        'R': {'name': 'Arginine', 'label': 15},
        'S': {'name': 'Serine', 'label': 16},
        'T': {'name': 'Threonine', 'label': 17},
        'V': {'name': 'Valine', 'label': 18},
        'W': {'name': 'Tryptophan', 'label': 19},
        'Y': {'name': 'Tyrosine', 'label': 20},
        '*': {'name': 'Stop', 'label': 0}
    }

    def __init__(self, aa):
        """
        Initialize the AminoAcid object with a single letter code.
        """
        if aa in self.properties:
            self.code = aa
            self.name = self.properties[aa]['name']
            self.label = self.properties[aa]['label']
        else:
            raise ValueError('Invalid amino acid code')

    def __str__(self):
        """
        Print the name of the amino acid.
        """
        return self.name

# Create an instance of the AminoAcid class
aa = AminoAcid('A')
print(aa)  # Prints: Alanine
print(aa.label)  # Prints: Ala