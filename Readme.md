# USPECS: US Presidential Election Candidate Speeches Corpus

## Overview

- **Project Aim:** Identify differences in thematic worldfields and sentiment in US presidentaial campaign speeches between 2007 an 2024, comparing Democrats and Republicans.
- **Used Methods**: Natural Language Processing, such as Topic Modelling and Sentiment Analysis 
- **Key Findings:** Differences in thematic worldfields across election periods and insights into rhethoric patterns based on the speaker.

## Dataset

- Speeches were extracted from the [American Presidency Project](https://www.presidency.ucsb.edu/) using the Python Library `beautifulsoup4`.
- The corpus contains 938 documents, 4.16 million tokens.
- Preprocessing was done using a piepline comprised of Tokenization, POS tagging, lemmatization, and Named Entity Recognition using additional Python libraries.

## Methodology

- **Techniques**
  - Topic Modelling: Using LDA with `Gensim`to extract thematic worldfields.
  - Sentiment Analysis: Using a lexicon-based approach with `NLTK`-Library (VADER-Sentiment Analysis) to extract sentiment-scores per sentence.
- **Libraries**
  - Main Python-Libraries: `spaCy`, `NLTK`, `SciPy` and  `Pandas`.
  - For visualization: `Plotly`, `Wordcloud` and `pyLDAvis`

## Getting started
`data`folder is not included in the repo instead it is attached as release.

Update the `data`:
- clone repo and download release
- zip release and add the `data` folder along the repo.
- remove rows of unwanted speeches either in the `metadata.csv` or the `metadata.xlsx`. inside the `data`folder.
- run `preprocessing/dataUpdater.py` once. It will delete the corresponding txt files and update the graphic.
- zip the updated `data` folder and create a new release (increment version and document the changes)
