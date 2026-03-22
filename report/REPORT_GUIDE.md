# Report Guide

Use this outline for the 6-10 page write-up.

## Suggested sections

1. Introduction
2. Problem statement and distinguisher objective
3. Cryptographic setup
4. Dataset generation process
5. Input representations
6. Model architectures
7. Experimental protocol
8. Results and analysis
9. Maximum effective round discussion
10. Limitations and future extensions

## Figures to include

- Pipeline diagram from plaintext pairs to labels
- Accuracy vs rounds
- ROC-AUC vs rounds
- Table comparing models and representations

## Key discussion points

- Why Speck32/64 was chosen
- Why each representation might reveal useful relations
- Which model generalizes best as rounds increase
- At what round count the distinguisher approaches chance
- How the random-permutation baseline was constructed
