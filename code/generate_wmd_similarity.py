# Source codes are: 
#https://github.com/zbmed-semtec/wmd-word2vec/blob/main/code/complete_relevance_matrix.py
#and
#https://github.com/zbmed-semtec/medline-preprocessing/blob/main/code/Cosine_Similarity/generate_cosine_existing_pairs.py
# This file includes the modifications to the source codes according to this project!

'''
Example
-------
To execute the script and generate document embeddings, you can run the following command:

python3 code/generate_wmd_similarity.py --input data/RELISH/Tokenized_Input/RELISH_Tokenized_Sample.npy -r data/relevance_w2v_blank.tsv -mod data/ -o data/w2v_relevance -c 18
    
'''
import argparse
import pandas as pd
import csv
import os

from gensim.models import KeyedVectors
from gensim import models

import numpy as np
import time

global_npy_dict = None
global_word2vec = None

    
def generate_npy_dict(filepath_in: str):
    '''
    Retrieves data from RELISH npy files, separating pmid and the document consisting of title and abstract..

    Parameters
    ----------
    filepath_in: str
        The filepath of the RELISH input npy file.
    Returns
    ----------
    dict of nump array
        A dictionary where each tokenized document is stored at their pmid.
    '''
    doc = np.load(filepath_in, allow_pickle=True)
    
    print('reading npy file')
    
    article_docs_dict = {}            
    for line in doc:
        
        # Check if the element is a list
        if isinstance(line[1], list):
            article_docs_dict[int(line[0])] = line[1] + line[2]
        else:
            document = np.ndarray.tolist(line[1])
            document.extend(np.ndarray.tolist(line[2]))
            article_docs_dict[int(line[0])] = [w for w in document]
            
    print('end of reading npy file')
    
    return article_docs_dict

    
def get_WMD_distance(pair: list):
    
    """
    Calculates WMD distance between two articles.
    Parameters
    ----------
    pair : list of str
        Tokenized document pair.
    Returns
    ----------
    float
        WMD distance.
    """
    return global_word2vec.wv.wmdistance(pair[0], pair[1])
 

def get_similarity_score(input_relevance_matrix: str, directory_model: str, output_matrix_name: str, iteration: int = -1) -> None:
    """
    Creates a 4 column matrix by appending cosine similarity scores for all existing pairs
    of PMIDs to the Relevance matrix.
    Parameters
    ----------
    input_relevance_matrix : str
        File path for relevance matrix (TREC/RELISH).
    directory_model : str
        File Path to the folder of all word2vec models.
    output_matrix_name : str
        File path for the generated 4 column matrix.
    iteration: int (optional)
        Nr of hyperparameter set: iteration is the suffix of the resulting matrix.
    """
    tokenset_pairs = []
    header = []
    rows = []
    with open(input_relevance_matrix, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        header = next(spamreader) # Save and remove header
        for row in spamreader:
            try:
                first_doc = global_npy_dict[int(row[0])]
                second_doc = global_npy_dict[int(row[1])]
                rows.append(row)
                tokenset_pairs.append((first_doc, second_doc))
            except:
                continue
    
    output_file = f"{output_matrix_name}_{iteration}.tsv"
       
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        #writer.writerow(header)
        writer.writerow(['PMID1', 'PMID2', 'Relevance', 'Similarity'])

        for ind, pair in enumerate(tokenset_pairs):
            rel_row = rows[ind]
            rel_row[3] = round(1./(1. + get_WMD_distance(pair)), 4)
            writer.writerow(rel_row)
  
    print('Saved matrix')

if __name__ == "__main__":
    __spec__ = None
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to input RELISH tokenized .npy file")
    parser.add_argument("-r", "--rel_matrix", type=str, help="Path of relevance matrix file")
    parser.add_argument("-mod", "--models_dir", type=str, help="File path for the folder containing models")
    parser.add_argument("-o", "--output", type=str, help="Output file path for generated 4 column wmd_distance matrix")
    parser.add_argument("-c", "--models_count", type=int, help="Number of word2vec models that have been created")
    args = parser.parse_args()
    
    global_npy_dict = generate_npy_dict(args.input)
    for iteration in range(args.models_count):
        print(f'start for set {iteration}')
        start = time.time()
        
        global_word2vec = KeyedVectors.load(f"{args.models_dir}/{iteration}/word2vec_model")
        
        get_similarity_score(args.rel_matrix, args.models_dir, args.output, iteration)
        print(f'done! for set {iteration} during {time.time() - start} seconds!')