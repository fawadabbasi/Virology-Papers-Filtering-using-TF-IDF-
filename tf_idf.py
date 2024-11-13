# Import libraries
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# Load dataset
data = pd.read_csv('collection_with_abstracts.csv')

# Initialize an NLP pipeline for embeddings
nlp_pipeline = pipeline('feature-extraction', model='bert-base-uncased')

# Keywords for filtering
keywords = [
    'CNN', 'convolutional neural network', 'RNN', 'recurrent neural network', 'LSTM', 
    'transformer', 'ResNet', 'U-Net', 'Graph Neural Network', 'GNN', 'Med-BERT',
    'neural network'
]

# Preprocessing and Filtering

def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\W+', ' ', text)  # remove punctuation
    return text

data['processed_abstract'] = data['abstract'].apply(lambda x: preprocess_text(x) if pd.notnull(x) else "")


# Filtering based on cosine similarity with keywords
def filter_papers(data, keywords, threshold=0.3):  # Adjusted threshold to 0.3
    vectorizer = TfidfVectorizer()
    all_text = data['processed_abstract'].tolist() + keywords
    tfidf_matrix = vectorizer.fit_transform(all_text)
    
    keyword_vectors = tfidf_matrix[-len(keywords):]
    text_vectors = tfidf_matrix[:-len(keywords)]

    relevance_scores = cosine_similarity(text_vectors, keyword_vectors).max(axis=1)
    data['relevance_score'] = relevance_scores
    
    # Print relevance scores for inspection
    print(data[['processed_abstract', 'relevance_score']].head(10))  # Print first 10 for inspection
    
    return data[data['relevance_score'] >= threshold]
filtered_data = filter_papers(data, keywords)

# Classify papers by method
def classify_method(text):
    text = text.lower()
    if any(term in text for term in ["text mining", "nlp", "natural language processing"]):
        return "text mining"
    elif any(term in text for term in ["image", "computer vision", "cnn", "convolutional"]):
        return "computer vision"
    elif any(term in text for term in ["text", "nlp", "natural language processing", "image", "cnn"]):
        return "both"
    else:
        return "other"

#filtered_data['method_type'] = filtered_data['processed_abstract'].apply(classify_method)
filtered_data.loc[:, 'method_type'] = filtered_data['processed_abstract'].apply(classify_method)


# Extract specific method names 
def extract_methods(text):
    methods = []
    patterns = [r'\bcnn\b', r'\brnn\b',r'\bgnn\b', r'\btransformer\b', r'\blstm\b', 
                r'\bconvolutional neural network\b',r'\bbert\b',r'\bMed-BERT\b',r'\bResNet\b']
    for pattern in patterns:
        if re.search(pattern, text):
            methods.append(re.search(pattern, text).group())
    return ", ".join(methods)

#filtered_data['extracted_methods'] = filtered_data['processed_abstract'].apply(extract_methods)
filtered_data.loc[:, 'extracted_methods'] = filtered_data['processed_abstract'].apply(extract_methods)

# print test result
# print(filtered_data)

# Save results to CSV
filtered_data[['PMID', 'method_type', 'extracted_methods']].to_csv('filtered_virology_papers.csv', index=False)