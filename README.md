# 442-2-Protein-cleavage
Projet 2 de INF442

We consider the problem of predicting the position of cleavage site in proteins, relying on patterns learned from a training set. Here a protein is known as its primary sequence, i.e. the sequence of its amino acids, which is given starting from the N-terminal side. Each amino acid is denoted by a letter code and, as there are 20 standard amino acids, plus some peculiarities, most of upper case letters are used in this encoding. Let A = {A, . . . , Z} be this alphabet.
For instance, the following sequence is the beginning of a protein, where the cleavage site is marked as the bond between the two underlined AR amino acids :

             MASKATLLLAFTLLFATCIARHQQRQQQQNQCQLQNIEALEPIEVIQAEA...

Then, for the purpose of this project, one will simply work on such sequences of letters. One has now to program some learning algorithms, to tune them, and to evaluate their performance. A simple statistical model, will be used first to establish a reference. One will then try to improve accuracy with some specific kernel functions for Support Vector Machines.


Ce qu'il reste a faire
tester le modele simple sur les neighboorhood de la base de données et verifier que l'algo les decrit bien comme tels
On peut considerer une base de données plus grande en mixant les 4 dataframes (avec concatenate (pas dur)) -> meme pas besoin, le fichier sig_13 est deja la concatenation des 3 fichiers
Attention peut etre un traitement necessaire comme pour le fichier 1

On peut changer la taille considérée du neighboorhood et prendre la taille maximale que l'on a a disposition

