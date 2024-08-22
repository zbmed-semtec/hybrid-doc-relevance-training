# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/utilities.py
# This file includes the modifications to the source codes according to this project!

import tqdm
import gensim
import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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
    return (pmids, article_docs)

# Create and train the Doc2Vec Model
def createDoc2VecModel(pmids: List[str], docs: List[List[str]], params: dict) -> Doc2Vec:
    """
    Create and train the Doc2Vec model using Gensim for the documents 
    in the corpus.

    Parameters
    ----------
    pmids: List[str]
            A list of all pubmed ids in the corpus.
    docs: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed document (title + abstract).
    params: dict
            Dictionary containing the parameters for the Doc2Vec model.
    Returns
    -------
    model: Doc2Vec
            Doc2Vec model.
    """
    tagged_data = [TaggedDocument(words=_d, tags=[str(pmids[i])])
                   for i, _d in enumerate(docs)]
    
    model = Doc2Vec(**params)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count,
                epochs=model.epochs)
    
    logging.info(f"Dataset vocabulary size: {model.wv.vectors.shape}")
    return model

# Save the Doc2Vec Model
def saveDoc2VecModel(model: Doc2Vec, output_file: str) -> None:
    """
    Saves the Doc2Vec model.

    Parameters
    ----------
    model: Doc2Vec
            Doc2Vec model.
    output_file: str
            File path of the Doc2Vec model generated.
    """
    model.save(output_file)

def calculate_cosine_similarity(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.

    This function computes the cosine similarity, which is defined as 1 minus the cosine distance 
    between two vectors. Cosine similarity is a measure of similarity between two non-zero vectors
    of an inner product space that measures the cosine of the angle between them.

    Parameters:
    ----------
    vector_1 : np.ndarray
        A numpy array representing the first vector.
    vector_2 : np.ndarray
        A numpy array representing the second vector.

    Returns:
    -------
    float
        The cosine similarity between vector_1 and vector_2.
    """
    return 1 - cosine(vector_1, vector_2)

def get_similarity_scores(input_relevance_matrix: str, embeddings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cosine similarity scores for pairs of PubMed IDs based on their embeddings and update a DataFrame with these scores.

    Parameters:
    ----------
    input_relevance_matrix : str
        File path to the TSV file containing pairs of PubMed IDs and a relevance value.
    embeddings_df : pd.DataFrame
        DataFrame containing PubMed IDs and their corresponding document embeddings.

    Returns:
    -------
    relevance_matrix_df : pd.DataFrame
        Updated DataFrame with cosine similarity scores added for each pair.
    """
    # 1) Read Relevance matrix
    column_names = ["PMID1", "PMID2", "Value"]
    relevance_matrix_df = pd.read_csv(input_relevance_matrix, sep="\t", names = column_names, skiprows=1)

    # 2) Adds empty columns to the file to store similarity scores
    relevance_matrix_df["Cosine Similarity"] = ""
    
    # 3) Create a dictionary to store embeddings
    embeddings_dict = {int(pmid): embedding for pmid, embedding in zip(embeddings_df['PMID'], embeddings_df['Embedding'])}

    # 4) Create a list of reference and assessed PMID pairs
    pmid_pairs = list(zip(relevance_matrix_df["PMID1"], relevance_matrix_df["PMID2"]))

    # 5) Calculate the cosine similarities between the document embeddings and update the relevance matrix dataframe
    for ref_pmid, assessed_pmid in tqdm.tqdm(pmid_pairs, total=len(pmid_pairs), desc="Calculating Similarities"):
        try:
            ref_pmid_vector = embeddings_dict[ref_pmid]
            assessed_pmid_vector = embeddings_dict[assessed_pmid]
            if ref_pmid_vector is not None and assessed_pmid_vector is not None:
                cosine_similarity = round(calculate_cosine_similarity(ref_pmid_vector, assessed_pmid_vector), 4)
                relevance_matrix_df.loc[(relevance_matrix_df['PMID1'] == ref_pmid) & (relevance_matrix_df['PMID2'] == assessed_pmid), 'Cosine Similarity'] = cosine_similarity
            else:
                print(f"One of the vectors is None for ({ref_pmid}, {assessed_pmid})")
                continue
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

def generate_embeddings(model: Doc2Vec, pmids: List[str], docs: List[List[str]]) -> pd.DataFrame:
    '''
    Generates document embeddings from a titles and abstracts in a given paper using doc2vec.

    Parameters
    ----------
    model: Doc2Vec
        Doc2Vec model
    pmids: list of str
        The list of all pmids which are processed.
    docs: list of list of str
        A two dimensional list of all tokenized article documents (title + abstract).
    Returns
    ----------
    embeddings_df: pd.Dataframe
        Pandas dataframe consisting of PMIDs and its corresponging embeddings.
    '''
    document_embeddings = []
    for doc in docs:
        # Infer vector for each document
        vector = model.infer_vector(doc)
        document_embeddings.append(vector)

    data = {"PMID": pmids, "Embedding": document_embeddings}
    embeddings_df = pd.DataFrame(data)
    embeddings_df = embeddings_df.sort_values("PMID")
    return embeddings_df

def save_embeddings_to_pickle(df: pd.DataFrame, output_file: str) -> None:
    """
    Save the DataFrame containing document embeddings to a pickle file.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing embeddings to be saved.
    output_file : str
        The file path where the DataFrame will be saved in pickle format.
    """
    df.to_pickle(output_file)