# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/utilities.py
# This file includes the modifications to the source codes according to this project!

import tqdm
import logging
import numpy as np
import pandas as pd
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from typing import Union, List
import ast  # This is used to convert the string representation of lists to actual lists

#--- To remove stopwords from the MeSH-terms in MeShIDtoPMID -----
#import nltk
#from nltk.corpus import stopwords

# Retrieves cleaned data from RELISH file
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

# Store test/validation Relish tokens in a dictionary with keys PMIDs after Post-processing the documnets' tokens to find 
# MeSH-terms in data and to append the corresponding MeSHIDs' to tokens
def generate_post_npy_dict_via_injection_MeSHIDs_into_tokens(filepath_in: str, MeShIDtoPMID: str) -> dict:
    '''
    Retrieves data from RELISH npy files, separating pmid and the document consisting of title and abstract.

    Parameters
    ----------
    filepath_in: str
        The filepath of the RELISH Validation/Test npy file.
    Returns
    ----------
    dict of nump array
        A dictionary where each tokenized document is stored at their pmid.
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
            
    logging.info('End of reading npy file & start reading TSV MeShIDtoPMID dict')
    
    # Now we should account for the newly inserted words
    
    # ---- We require this for situations where the input tokens have had stopwords removed -----------
    #nltk.download('stopwords')
    #stop_words = set(stopwords.words('english'))
    #--------------------------------------------------
    
    # Read the TSV file into a DataFrame
    df = pd.read_csv(MeShIDtoPMID, sep='\t', header=None, names=['MeSHID', 'Appearance(pmid , tokenized lowercase words)'], skiprows=1)

    # Convert the string representation of lists to actual lists using ast.literal_eval
    df['Appearance(pmid , tokenized lowercase words)'] = df['Appearance(pmid , tokenized lowercase words)'].apply(ast.literal_eval)
    
    for meshID, all_with_mesh_term in zip(df['MeSHID'], df['Appearance(pmid , tokenized lowercase words)']):
        for pmid_term in all_with_mesh_term:
            article_with_MeSHterm = int(pmid_term[0])
            try:
                #----------------------------------------------------------------
                pattern_to_find = [w for w in pmid_term[1:] if not w in stop_words]  # removal of stopwords from MeSH-term
                # Find indices of the pattern in the list
                indices = [i for i in range(len(article_docs_dict[article_with_MeSHterm]) - len(pattern_to_find) + 1) 
                           if article_docs_dict[article_with_MeSHterm][i:i+len(pattern_to_find)] == pattern_to_find]
                # Iterate over the indices in reverse order
                for index in reversed(indices):
                    # Insert MeSHID at the beginning of the pattern
                    article_docs_dict[article_with_MeSHterm].insert(index, str(meshID).lower())
            except:
                continue
    with open('dict.csv', 'w') as file:
            for key in article_docs_dict:
                file.write("%s,%s\n"%(key,article_docs_dict[key]))
    return article_docs_dict

# Create and train the Word2Vec Model
def createWord2VecModel(pmids: List[str], article_docs: List[List[str]], params: dict) -> Word2Vec:
    """
    Create and train the Word2Vec model using Gensim for the documents' tokens 
    in the corpus.

    Parameters
    ----------
    pmids: List[str]
            A list of all pubmed ids in the corpus.
    docs: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed document (title + abstract).
    params: dict
            Dictionary containing the parameters for the Word2Vec model.
    Returns
    -------
    model: Word2Vec
            Word2Vec model.
    """
    
    sentence_list = []
    for index in range(len(pmids)):
        sentence_list.append(article_docs[index])
    params['sentences'] = sentence_list
    
    model = Word2Vec(**params)

    return model
    
# Finding MeSH-terms in training' tokens to calculate the corresponding MeSHIDs' embeddings and append them to the generated model
def injection_MeSHembeddings_into_embeddings(model: Word2Vec, pmids: str, article_doc: list, MeShIDtoPMID: str):
    '''
    Using the generated word embeddings and MeShIDtoPMID tsv-file, compute the centroid of embeddings corresponding to each MeSHID
    in training data and append the computed centroid to the list of all word embeddings of the corresponding articles.
    
    Parameters
    ----------
    model: Word2Vec
        Word2Vec model.
    pmids: list of str
        The list of all pmids which are processed.
    article_doc_global: list of list of str
        A two dimensional list of all tokenized article documents (title + abstract).
    MeShIDtoPMID
        File path for the tsv file whose rows consist of MeSHIDs and lists of [PMID, words].
    '''
    batch_embeddings_MeSHID = []
    batch_meshIDs = []

    # Read the TSV file into a DataFrame
    df = pd.read_csv(MeShIDtoPMID, sep='\t', header=None, names=['MeSHID', 'Appearance(pmid , tokenized lowercase words)'], skiprows=1)

    # Convert the string representation of lists to actual lists using ast.literal_eval
    df['Appearance(pmid , tokenized lowercase words)'] = df['Appearance(pmid , tokenized lowercase words)'].apply(ast.literal_eval)
    
    for meshID, all_with_mesh_term in zip(df['MeSHID'], df['Appearance(pmid , tokenized lowercase words)']):
        counter_article = 0
        embeddings_MeSHID = np.zeros(model.vector_size, dtype = float)
        for pmid_term in all_with_mesh_term:
            article_with_MeSHterm = int(pmid_term[0])
            if article_with_MeSHterm in pmids:
                iteration = pmids.index(article_with_MeSHterm)
                embeddings_set_of_terms = np.zeros(model.vector_size, dtype = float)
                counter_terms = 0
                for word in pmid_term[1:]:
                    try:
                        embeddings_set_of_terms += model.wv[word]
                        counter_terms += 1
                    except:
                        continue

                if counter_terms:
                    embeddings_set_of_terms /= counter_terms
                    embeddings_MeSHID += embeddings_set_of_terms
                    counter_article += 1

        if counter_article:
            embeddings_MeSHID /= counter_article
            #model.wv.add_vector(str(meshID).lower(), embeddings_MeSHID)
            batch_embeddings_MeSHID.append(embeddings_MeSHID)
            batch_meshIDs.append(str(meshID).lower())
                
    # Add embedding-vectors in batches: Unfortunately using this makes the saved model unloadable!
    #model.wv.add_vectors(batch_meshIDs, batch_embeddings_MeSHID, replace  = True)
    
    '''
    Preallocating space for the required size means allocating memory for the entire set of vectors upfront, rather than 
    dynamically resizing the storage as vectors are added. This can help avoid the inefficiency of adding vectors one by one.
    '''
    # Define the number of new embedding-vectors that must be added to the trained model
    num_new_vectors = len(batch_embeddings_MeSHID)
    # Dimensionality of the vectors
    vector_size = model.vector_size
    # Get the current number of vectors in the model
    current_num_vectors = len(model.wv.vectors)
    # Calculate the total number of vectors after adding the new ones
    total_num_vectors = current_num_vectors + num_new_vectors

    # Preallocate space for vectors with zeros
    preallocated_vectors = np.zeros((total_num_vectors, vector_size), dtype=np.float32)

    # Copy the existing vectors to the preallocated array
    preallocated_vectors[:current_num_vectors] = model.wv.vectors

    # Add the new vectors to the preallocated array
    for i in range(num_new_vectors):
        new_embedding = batch_embeddings_MeSHID[i]
        new_meshID = batch_meshIDs[i]
        preallocated_vectors[current_num_vectors + i] = new_embedding
        # Update the vocab dictionary with the new meshID and assigning a proper integer index to it
        model.wv.key_to_index[new_meshID] = current_num_vectors + i

    # Update the model's vectors with the preallocated array
    model.wv.vectors = preallocated_vectors
    
    return model  

# Post-processing of the documnets' tokens to find MeSH-terms in test/validation data and to append the corresponding MeSHIDs to tokens
def injection_MeSHIDs_into_tokens(pmids: str, article_doc: list, MeShIDtoPMID: str):
    '''
    Using the generated word embeddings and MeShIDtoPMID tsv-file, append MeSHIDs as new words to the list of tokens of the 
    corresponding articles containing the corresponding MeSH-terms.
    
    Parameters
    ----------
    pmids: list of str
        The list of all test/validation pmids which are processed.
    article_doc_global: list of list of str
        A two dimensional list of all tokenized test/validation article documents (title + abstract).
    MeShIDtoPMID: str
        File path for the tsv file whose rows consist of MeSHIDs and lists of [PMID, words].
    '''

    # Read the TSV file into a DataFrame
    df = pd.read_csv(MeShIDtoPMID, sep='\t', header=None, names=['MeSHID', 'Appearance(pmid , tokenized lowercase words)'], skiprows=1)

    # Convert the string representation of lists to actual lists using ast.literal_eval
    df['Appearance(pmid , tokenized lowercase words)'] = df['Appearance(pmid , tokenized lowercase words)'].apply(ast.literal_eval)
    
    for meshID, all_with_mesh_term in zip(df['MeSHID'], df['Appearance(pmid , tokenized lowercase words)']):
        for pmid_term in all_with_mesh_term:
            article_with_MeSHterm = int(pmid_term[0])
            #if article_with_MeSHterm in pmids:
            try:
                iteration = pmids.index(article_with_MeSHterm)
                
                pattern_to_find = [w for w in pmid_term[1:] if not w in stop_words] #removal of stopwords from MeSH-term
                # Find indices of the pattern in the list
                indices = [i for i in range(len(article_doc[iteration]) - len(pattern_to_find) + 1) 
                           if article_doc[iteration][i:i+len(pattern_to_find)] == pattern_to_find]
                # Iterate over the indices in reverse order
                for index in reversed(indices):
                    # Insert MeSHID at the beginning of the pattern
                    article_doc[iteration].insert(index, str(meshID).lower())
            except:
                continue
                
    return article_doc

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

def get_WMD_similarity_scores(input_relevance_matrix: str, model: Word2Vec,  article_post_annot_docs_dict: dict) -> pd.DataFrame:
    """
    Creates a 4 column matrix by appending WMD scores for all existing pairs
    of PMIDs to the Relevance matrix.
    Parameters
    ----------
    input_relevance_matrix : str
        File path for RELISH relevance matrix.
    model: Word2Vec
        Word2Vec model.
    article_post_annot_docs_dict: dict
        Post-annotated test/validation tokens in a dictionary with keys PMIDs 
    output_matrix_name : str
        File path for the generated 4 column matrix.
    """
    # 1) Read Relevance matrix
    column_names = ["PMID1", "PMID2", "Value"]
    relevance_matrix_df = pd.read_csv(input_relevance_matrix, sep="\t", names = column_names, skiprows=1)
    
    # 2) Adds empty columns to the file to store similarity scores
    relevance_matrix_df["WMD"] = ""
    
    # 3) Create a list of ref and assessed PMID-pairs and their relevance-scores
    pmid_pairs_relevance = list(zip(relevance_matrix_df["PMID1"], relevance_matrix_df["PMID2"], relevance_matrix_df["Value"]))
    
    # 4) Calculate the WMD between the document embeddings and update the relevance matrix dataframe
    for ref_pmid, assessed_pmid, rel_value in tqdm.tqdm(pmid_pairs_relevance, total=len(pmid_pairs_relevance), 
                                                        desc="Calculating WMD Similarities"):
        try:
            ref_doc = article_post_annot_docs_dict[int(ref_pmid)]
            assessed_doc = article_post_annot_docs_dict[int(assessed_pmid)]
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

def loadModel(model_path: str) -> None:
    """
    Loads the saved Word2Vec model.

    Parameters
    ----------
    model_path: str
            Path of the Word2Vec model.

    Return
    ----------
    model: Word2Vec
            WordVec model.
    """
    model = gensim.models.Word2Vec.load(model_path)
    return model