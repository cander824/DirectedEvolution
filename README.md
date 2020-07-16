# DirectedEvolution

Protein data taken from https://www.protabank.org/study_analysis/UGoTemyw/

This function builds a CNN using the mutation library given in the study above.
Then, it generates a bunch of mutants with different combinations of the various mutations from the training set.
Finally, it evaluates all of these in silico mutants with the CNN and finds the optimal mutant.

Currently, this only performs one round of directed evolution. 
