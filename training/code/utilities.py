# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/utilities.py
# This file includes the modifications to the source codes according to this project!

import tqdm
import gensim
import numpy as np
import pandas as pd
import csv
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec
from typing import Union, List
import gensim.models as model
import ast  # This is used to convert the string representation of lists to actual lists

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
            
    print('end of reading npy file and save it as dictionary with keys PMIDs')
    
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
            model.wv.add_vector(str(meshID).lower(), embeddings_MeSHID)
                
    return model  

# Post-processing of the documnets' tokens to find MeSH-terms in test/validation data and to append the corresponding MeSHIDs' to tokens
def injection_MeSHIDs_into_tokens(model: Word2Vec, pmids: str, article_doc: list, global_article_Annot_docs_dict: dict,
                                  MeShIDtoPMID: str, reduction: int):
    '''
    Using the generated word embeddings and MeShIDtoPMID tsv-file, append MeSHIDs as new words to the list of tokens of the 
    corresponding articles containing the corresponding MeSH-terms.
    
    Parameters
    ----------
    model: Word2Vec
            Word2Vec model.
    pmids: list of str
        The list of all test/validation pmids which are processed.
    article_doc_global: list of list of str
        A two dimensional list of all tokenized test/validation article documents (title + abstract).
    global_article_Annot_docs_dict: dict
        Store annotated test/validation tokens in a dictionary with keys PMIDs
    MeShIDtoPMID: str
        File path for the tsv file whose rows consist of MeSHIDs and lists of [PMID, words].
    reduction: int
        Whether to reduce the documents'words by replacing the catalogued ones with corresponding MeSHID (1) or not (0).
    '''

    # Read the TSV file into a DataFrame
    df = pd.read_csv(MeShIDtoPMID, sep='\t', header=None, names=['MeSHID', 'Appearance(pmid , tokenized lowercase words)'], skiprows=1)

    # Convert the string representation of lists to actual lists using ast.literal_eval
    df['Appearance(pmid , tokenized lowercase words)'] = df['Appearance(pmid , tokenized lowercase words)'].apply(ast.literal_eval)
    
    for meshID, all_with_mesh_term in zip(df['MeSHID'], df['Appearance(pmid , tokenized lowercase words)']):
        ArticlesList_with_MeSHterm = []
        for pmid_term in all_with_mesh_term:
            article_with_MeSHterm = int(pmid_term[0])
            if article_with_MeSHterm in pmids:
                iteration = pmids.index(article_with_MeSHterm)
                if reduction:
                    for word in pmid_term[1:]:
                        try:
                            if word not in global_article_Annot_docs_dict[article_with_MeSHterm]:
                                article_doc[iteration].remove(word)
                        except:
                            continue

                ArticlesList_with_MeSHterm.append(iteration) #So MeSHembeddings will be injected to all articles with corresponding MeSHIDs

        for itr in ArticlesList_with_MeSHterm:
            article_doc[itr].append(str(meshID).lower())
                
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

def calculate_cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def get_similarity_scores(input_relevance_matrix, embeddings, output_matrix_name):
    # Read Embeddings
    embeddings_df = pd.read_pickle(embeddings)
    
    # Read Relevance matrix
    column_names = ["PID1", "PID2", "Value"]
    relevance_matrix_df = pd.read_csv(input_relevance_matrix, sep="\t", names = column_names, skiprows=1)

    # Adds empty columns to the file to store similarity scores
    relevance_matrix_df["Cosine Similarity"] = ""

    #print(relevance_matrix_df)

    # Create a dictionary to store embeddings
    embeddings_dict = {pmid: embedding for pmid, embedding in zip(embeddings_df['PID'], embeddings_df['Embedding'])}

    # Create a list of ref and assessed PMID pairs
    pmid_pairs = list(zip(relevance_matrix_df["PID1"], relevance_matrix_df["PID2"]))

    for ref_pmid, assessed_pmid in tqdm.tqdm(pmid_pairs, total=len(pmid_pairs), desc="Calculating Similarities"):
        try:
            ref_pmid_vector = embeddings_dict[ref_pmid]
            assessed_pmid_vector = embeddings_dict[assessed_pmid]
            if ref_pmid_vector is not None and assessed_pmid_vector is not None:
                cosine_similarity = round(calculate_cosine_similarity(ref_pmid_vector, assessed_pmid_vector), 4)
                relevance_matrix_df.loc[(relevance_matrix_df['PID1'] == ref_pmid) & (relevance_matrix_df['PID2'] == assessed_pmid), 
                                        'Cosine Similarity'] = cosine_similarity
            else:
                print(f"One of the vectors is None for ({ref_pmid}, {assessed_pmid})")
        except KeyError as e:
            print(f"\nKeyError: {e}, ref_pmid: {ref_pmid}, assessed_pmid: {assessed_pmid}")
            #break
            continue

    print('Added similarity scores')
    
    # Saves the updated matrix 
    relevance_matrix_df.to_csv(output_matrix_name, index=False, sep="\t")
    print('Saved matrix')

def generate_embeddings(model, pmids, docs, output_file):
    embeddings_list = []
    for doc in docs:
        # Infer vector for each document
        vector = model.infer_vector(doc)
        embeddings_list.append(vector)
    save_embeddings_to_pickle(pmids, embeddings_list, output_file)

def save_embeddings_to_pickle(pmids, embeddings_list, output_file):
    data = {"PID": pmids, "Embedding": embeddings_list}
    df = pd.DataFrame(data)
    df = df.sort_values("PID")
    df.to_pickle(output_file)
    print(f"Embeddings saved to {output_file}")

# Generate document embeddings using the centroid of word embeddings
def generate_document_embeddings(model: Word2Vec, pmids: str, article_doc: list, output_file: str):
    '''
    Generates document embeddings from a titles and abstracts in a given paper using word2vec and calculating the cenroids of all given word embeddings.
    If no gensim model is given, the 'glove-wiki-gigaword-200' gensim model is used.
    
    Parameters
    ----------
    model: Word2Vec
        Word2Vec model.
    pmids: list of str
        The list of all pmids which are processed.
    article_doc: list of list of str
        A two dimensional list of all tokenized article documents (title + abstract).
    directory_out: str
        The filepath of the output directory of all .npy embeddings.
    '''

    missing_words = 0
    iteration = 0
    document_embeddings = []
    for iteration in range(len(pmids)):
        # Retrieve word embeddings.
        embedding_list = []
        for word in article_doc[iteration]:
            try:
                embedding_list.append(model.wv[word])
            except:
                missing_words += 1
       

        # Generate document embeddings from word embeddings using word-vector centroids.
        if len(embedding_list) == 0:
            # This can be caused by a high min-count parameter or missing vocabulary when using a pretrained model
            document_embeddings.append([])
            continue
        document = [0.0] * model.vector_size

        for dim in range(model.vector_size):
            for word_embeddings in embedding_list:
                document[dim] += word_embeddings[dim]
            document[dim] = document[dim] / len(embedding_list)
        document_embeddings.append(document)
        
    save_embeddings_to_pickle(pmids, document_embeddings, output_file)