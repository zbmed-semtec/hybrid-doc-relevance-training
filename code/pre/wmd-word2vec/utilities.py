# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/utilities.py
# This file includes the modifications to the source codes according to this project!

import tqdm
import gensim
import logging
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from typing import Union, List

# Retrieves cleaned data from RELISH and TREC npy files
def process_data_from_npy(file_path_in: str = None) -> Union[List[str], List[List[str]], List[List[str]], List[List[str]]]:
    """
    Retrieves cleaned data from RELISH and TREC npy files, separating each column 
    into their own respective list.

    Parameters
    ----------
    filepathIn: str
            The filepath of the RELISH or TREC input npy file.
    Returns
    -------
    pmids: List[str]
            A list of all pubmed ids in the corpus.
    titles: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed title.
    abstracts: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed abstract.
    docs: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed document (title + abstract).
    """
    doc = np.load(file_path_in, allow_pickle=True)
    pmids = []
    article_docs = []
    
    for line in range(len(doc)):
        pmids.append(int(doc[line][0]))
        
        # Check if the element is a NumPy array before using tolist
        if isinstance(doc[line][1], np.ndarray):
            article_docs.append(doc[line][1].tolist())
        else:
            article_docs.append(doc[line][1])
        
        # Check if the element is a NumPy array before using tolist
        if isinstance(doc[line][2], np.ndarray):
            article_docs[line].extend(doc[line][2].tolist())
        else:
            article_docs[line].extend(doc[line][2])
    return pmids, article_docs

# Store Relish tokens in a dictionary with keys PMIDs
def generate_npy_dict(filepath_in: str=None)->dict:
    '''
    Retrieves data from RELISH npy files, separating pmid and the document consisting of title and abstract..

    Parameters
    ----------
    filepath_in: str
        The filepath of the RELISH input npy file.
    Returns
    ----------
    list of str
        All pubmed ids associated to the paper.
    list of list of str
        All tokenized words within the preprocessed title + abstract.
    '''
    doc = np.load(filepath_in, allow_pickle=True)
    
    logging.info('Reading npy file')
    
    article_docs_dict = {}            
    for line in doc:
        
        # Check if the element is a list
        if isinstance(line[1], list):
            article_docs_dict[int(line[0])] = line[1] + line[2]
        else:
            document = np.ndarray.tolist(line[1])
            document.extend(np.ndarray.tolist(line[2]))
            article_docs_dict[int(line[0])] = [w for w in document]
            
    logging.info('End of reading npy file and save it as dictionary with keys PMIDs')
    
    return article_docs_dict

# Create and train the Word2Vec Model
def generate_Word2Vec_model(pmids: list, article_doc: list, params: dict) -> Word2Vec:
    '''
    Generates a word2vec model from all RELISH or TREC sentences using gensim and saves it as a .model file.

    Parameters
    ----------
    pmids: list of str
        A list of all appearing pubmed ids in the input dataset.
    article_doc: list of list of str
        A two dimensional list of all tokenized article documents (title + abstract).
    params: dict
        A dictionary of the hyperparameters for the model.
    Returns
    -------
    model: Word2Vec
            Word2Vec model.
    '''
    sentence_list = []
    for index in range(len(pmids)):
        sentence_list.append(article_doc[index])
    params['sentences'] = sentence_list
    wv_model = None
    wv_model = Word2Vec(**params)

    logging.info(f"Dataset vocabulary size: {wv_model.wv.vectors.shape}")
    return wv_model

# Save the Word2Vec Model
def saveWord2VecModel(model: Word2Vec, output_file: str) -> None:
    """
    Saves the Word2Vec model.

    Parameters
    ----------
    model: Word2Vec
            Word2Vec model.
    output_file: str
            File path of the Word2Vec model generated.
    """
    model.save(output_file)

def get_WMD_distance(model: Word2Vec, document1: list, document2: list):
    
    """
    Calculates WMD distance between two articles.
    Parameters
    ----------
    model: Word2Vec
            Word2Vec model.
    document1 : list of str
        Tokenized document.
    document2 : list of str
        Tokenized document.
    Returns
    ----------
    float
        WMD distance.
    """
    return model.wv.wmdistance(document1, document2)

def get_WMD_similarity_scores(input_relevance_matrix: str, model: Word2Vec, docs_dict:dict) -> pd.DataFrame:
    """
    Calculate word mover's distance for pairs of PubMed IDs based on their embeddings and update a DataFrame with these scores.

    Parameters:
    ----------
    input_relevance_matrix : str
        File path to the TSV file containing pairs of PubMed IDs and a relevance value.
    embeddings_df : pd.DataFrame
        DataFrame containing PubMed IDs and their corresponding document embeddings.
    docs_dict: dict
        Test/validation tokens in a dictionary with keys PMIDs 
    Returns:
    -------
    relevance_matrix_df : pd.DataFrame
        Updated DataFrame with cosine similarity scores added for each pair.
    """
    # 1) Read Relevance matrix
    column_names = ["PMID1", "PMID2", "Value"]
    relevance_matrix_df = pd.read_csv(input_relevance_matrix, sep="\t", names = column_names, skiprows=1)

    # 2) Adds empty columns to the file to store similarity scores
    relevance_matrix_df["WMD"] = ""
    
    # 3) Create a dictionary to store embeddings
    # embeddings_dict = {int(pmid): embedding for pmid, embedding in zip(embeddings_df['PMID'], embeddings_df['Embedding'])}

    # 4) Create a list of reference and assessed PMID pairs
    pmid_pairs = list(zip(relevance_matrix_df["PMID1"], relevance_matrix_df["PMID2"]))

    # 5) Calculate the cosine similarities between the document embeddings and update the relevance matrix dataframe
    for ref_pmid, assessed_pmid in tqdm.tqdm(pmid_pairs, total=len(pmid_pairs), desc="Calculating Similarities"):
        try:
            ref_doc = docs_dict[int(ref_pmid)]
            assessed_doc = docs_dict[assessed_pmid]
            WMD_similarity = round(1./(1. + get_WMD_distance(model, ref_doc, assessed_doc) ), 4)
            relevance_matrix_df.loc[(relevance_matrix_df['PMID1'] == ref_pmid) & (relevance_matrix_df['PMID2'] == assessed_pmid), 'WMD'] = WMD_similarity
        except KeyError as e:
            print(f"\nKeyError: {e}, ref_pmid: {ref_pmid}, assessed_pmid: {assessed_pmid}")

    return relevance_matrix_df


def save_similarity_to_tsv(df: pd.DataFrame, output_file: str) -> None:
    """
    Save the DataFrame containing similarity scores to a TSV file.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame to be saved, containing similarity scores among other data.
    output_file : str
        The file path where the DataFrame will be saved as a TSV.
    """
    df.to_csv(output_file, index=False, sep="\t")