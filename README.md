# Virology Papers Filtering and Classification using TF-IDF and Cosine Similarity

This project processes and classifies research papers on virology based on relevant keywords and their similarity to scientific concepts. By leveraging TF-IDF and cosine similarity, the project filters abstracts and categorizes papers into predefined research areas based on relevance.

## Project Overview

The code performs the following:
1. **Data Preprocessing:** Cleans text by converting it to lowercase and removing punctuation.
2. **Filtering by Similarity:** Uses TF-IDF vectors and cosine similarity to filter papers based on their relevance to predefined keywords.
3. **Classification by Research Area:** Classifies papers as "text mining," "computer vision," "both," or "other" based on content.
4. **Method Extraction:** Extracts specific research methods (e.g., "CNN," "RNN," "transformer") mentioned in each paper for further analysis.

## Solution Components

### 1. Data Preprocessing

- The abstract text is preprocessed by:
  - Converting it to lowercase.
  - Removing punctuation to ensure consistency for keyword matching.

### 2. Filtering Using Cosine Similarity with TF-IDF

The project calculates the similarity between the abstracts and predefined **keywords** using **TF-IDF** vectors:
- **TF-IDF Vectorization:** The `TfidfVectorizer` is used to create vectors for all abstracts and keywords, capturing important terms by accounting for term frequency and inverse document frequency.
- **Cosine Similarity Calculation:** Cosine similarity scores between abstracts and keywords determine relevance.
- **Threshold Filtering:** Only papers with a relevance score above a threshold (0.3) are retained, helping to eliminate unrelated content.

### 3. Classification of Research Area

Each paper is classified based on content:
- **Text Mining/NLP:** Papers mentioning terms like "NLP," "text mining," or "natural language processing."
- **Computer Vision:** Papers related to "image processing," "CNN," or "computer vision."
- **Both:** Papers discussing "text mining" and "computer vision."
- **Other:** Papers that do not fit the above categories.

### 4. Method Extraction

Specific research methods are extracted from each paper using regular expressions. Methods such as "CNN," "RNN," "transformer," and "Med-BERT" are identified and listed for each abstract.

## Requirements

To install the required dependencies, use:

```bash
pip install pandas numpy scikit-learn transformers
```
## Why idf is better approach?
Keyword-based filtering treats each keyword equally and checks if a term exists in the text, regardless of its importance or frequency.
TF-IDF calculates term importance based on how often a term appears in a document relative to its occurrence across all documents. Terms frequently appearing in one document (high term frequency) but rare across the dataset (high inverse document frequency) are given higher weights, highlighting unique and relevant terms over common, generic words.
