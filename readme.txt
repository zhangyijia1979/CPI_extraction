Chemical-protein interaction extraction via contextualized word representations and multi-head attention
================================================================================

Chemical-protein interaction extraction aims to detect and classify the Chemical-protein relation from the free biomedical text, which is developed with Keras using Tensorflow as the backend. This package contains a demo implementation of CPI extraction described in the paper "Chemical-protein interaction extraction via contextualized word representations and multi-head attention
."

This is research software, provided as is without express or implied warranties etc. see licence.txt for more details. We have tried to make it reasonably usable and provided help options, but adapting the system to new environments or transforming a corpus to the format used by the system may require significant effort. 

The details of related files are described as follows. ChemProt directory contains the original ChemProt Corpus. “CPI_extraction.py“ is the source code of the demo implementation. “chemprot_train.pkl”, “chemprot_development.pkl” and “chemprot_test.pkl” are the proprocessed ChemProt training, development and testing data, respectively.


============================ QUICKSTART ========================================

The demo implementation has tested on Keras 2.0.2, python 3.5, Tensorflow 1.8 and Tensorflow-hub 0.1.1.

User can use CPI_extraction.py to automatic extract CPIs from the processed pkl files.

