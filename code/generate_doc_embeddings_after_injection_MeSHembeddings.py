# Source code: https://github.com/zbmed-semtec/word2doc2vec-doc-relevance/blob/main/code/generate_embeddings.py
# This file includes the modifications to the source code according to this project

'''
Example
-------
To execute the script and generate document embeddings, you can run the following command which includes reduction:

python3 code/generate_doc_embeddings_after_injection_MeSHembeddings.py --input data/RELISH/Tokenized_Input/RELISH_Tokenized_Sample.npy -annoti data/RELISH/Tokenized_Input/RELISH_Annot_Tokens_Sample.npy --output data/ --params_json data/hyperparameters_word2vec.json --use_pretrained 0 --MeShIDtoPMID data/dic_MeShIDtoPMID_2022628.tsv -rd 1
    
'''

import argparse
import json
import numpy as np
import time

global_article_Annot_docs_dict = None

def prepare_from_npy(filepath_in: str):
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
            
    print('end of reading npy file')
        
    return (pmids, article_docs)


def generate_npy_dict(filepath_in: str):
    '''
    Retrieves data from RELISH npy files, separating pmid and the document consisting of title and abstract..

    Parameters
    ----------
    filepath_in: str
        Path to input Annotated RELISH tokenized .npy file
    Returns
    ----------
    list of str
        All pubmed ids associated to the paper.
    list of list of str
        All tokenized words within the preprocessed title + abstract.
    '''
    doc = np.load(filepath_in, allow_pickle=True)
    
    print('reading Annotated npy file')
    
    article_docs_dict = {}            
    for line in doc:
        
        # Check if the element is a list
        if isinstance(line[1], list):
            article_docs_dict[int(line[0])] = line[1] + line[2]
        else:
            document = np.ndarray.tolist(line[1])
            document.extend(np.ndarray.tolist(line[2]))
            article_docs_dict[int(line[0])] = [w for w in document]
            
    print('end of reading Annotated npy file')
    
    return article_docs_dict


def generate_Word2Vec_model(article_doc: list, pmids: list, params: list, filepath_out: str, use_pretrained: bool):
    '''
    Generates a word2vec model from all RELISH sentences using gensim and saves it as a .model file.

    Parameters
    ----------
    article_doc: list of list of str
        A two dimensional list of all tokenized article documents (title + abstract).
    pmids: list of str
        A list of all appearing pubmed ids in the input dataset.
    params: dict
        A dictionary of the hyperparameters for the model.
    filepath_out: str
        The filepath for the resulting word2vec model file.
    use_pretrained: bool
        Whether to use a pretrained Word2Vec model.
    '''
    from gensim.models import Word2Vec
    sentence_list = []
    for index in range(len(pmids)):
        sentence_list.append(article_doc[index])
    params['sentences'] = sentence_list
    wv_model = None
    if use_pretrained:
        print("Pretraining")
    else:
        wv_model = Word2Vec(**params)
    wv_model.save(filepath_out)
    

def injection_MeSHembeddings_into_embeddings(pmids: str, article_doc: list, MeShIDtoPMID: str, directory_out: str,
                                             param_iteration: int, reduction: int, gensim_model_path: str = ""):
    '''
    Using the generated word embeddings and MeShIDtoPMID tsv-file, compute the centroid of embeddings corresponding to each MeSHID and
    append the computed centroid to the list of all word embeddings of the corresponding articles.
    
    Parameters
    ----------
    pmids: list of str
        The list of all pmids which are processed.
    article_doc_global: list of list of str
        A two dimensional list of all tokenized article documents (title + abstract).
    MeShIDtoPMID
        File path for the tsv file whose rows consist of MeSHIDs and lists of [PMID, words].
    directory_out: str
        The filepath of the output directory of all .npy embeddings.
    param_iteration: int
        Iteration through paramter list.
    reduction: int
        Whether to reduce the documents'words by replacing the catalogued ones with corresponding MeSHID (1) or not (0).
    gensim_model_path: str (optional)
        The filepath of the custom gensimModel.
    '''
    import gensim.downloader as api
    import gensim.models as model
    import csv
 
    import os
    import pandas as pd
    import ast  # This is used to convert the string representation of lists to actual lists
        
    word_vectors = None
    has_custom_model = gensim_model_path != ""
    if has_custom_model:
        word_vectors = model.Word2Vec.load(gensim_model_path)
    else:
        print('using pretrained model')
        word_vectors = api.load('word2vec-google-news-300')

    # Read the TSV file into a DataFrame
    df = pd.read_csv(MeShIDtoPMID, sep='\t', header=None, names=['MeSHID', 'Appearance(pmid , tokenized lowercase words)'], skiprows=1)

    # Convert the string representation of lists to actual lists using ast.literal_eval
    df['Appearance(pmid , tokenized lowercase words)'] = df['Appearance(pmid , tokenized lowercase words)'].apply(ast.literal_eval)
    
    
    for meshID, all_with_mesh_term in zip(df['MeSHID'], df['Appearance(pmid , tokenized lowercase words)']):
        counter_article = 0
        embeddings_MeSHID = np.zeros(word_vectors.vector_size, dtype = float)
        ArticlesList_with_MeSHterm = []
        for pmid_term in all_with_mesh_term:
            article_with_MeSHterm = int(pmid_term[0])
            if article_with_MeSHterm in pmids:
                iteration = pmids.index(article_with_MeSHterm)
                embeddings_set_of_terms = np.zeros(word_vectors.vector_size, dtype = float)
                counter_terms = 0
                for word in pmid_term[1:]:
                    try:
                        embeddings_set_of_terms += word_vectors.wv[word]
                        counter_terms += 1

                        if reduction:
                            # to account for individual words of a MeSH term which also appear independently in the text.
                            if word not in global_article_Annot_docs_dict[article_with_MeSHterm]:
                                article_doc[iteration].remove(word)

                    except:
                        continue

                if counter_terms:
                    embeddings_set_of_terms /= counter_terms
                    embeddings_MeSHID += embeddings_set_of_terms
                    counter_article += 1

                ArticlesList_with_MeSHterm.append(iteration) #So MeSHembeddings will be injected to all articles with corresponding MeSHIDs
                
        if counter_article:
            embeddings_MeSHID /= counter_article
            word_vectors.wv.add_vector(str(meshID), embeddings_MeSHID)
            for itr in ArticlesList_with_MeSHterm:
                article_doc[itr].append(str(meshID))
                
    # Save the updated model
    word_vectors.save(gensim_model_path)
    
    os.makedirs(f"{directory_out}/{param_iteration}", exist_ok=True)
    word_vectors.save(f"{directory_out}/{param_iteration}/word2vec_model")

    
def generate_document_embeddings(pmids: str, article_doc: list, directory_out: str, param_iteration: int, gensim_model_path: str = ""):
    '''
    Generates document embeddings from a titles and abstracts in a given paper using word2vec and calculating the cenroids of all given word embeddings.
    If no gensim model is given, the 'glove-wiki-gigaword-200' gensim model is used.
    
    Parameters
    ----------
    pmids: list of str
        The list of all pmids which are processed.
    article_doc: list of list of str
        A two dimensional list of all tokenized article documents (title + abstract).
    directory_out: str
        The filepath of the output directory of all .npy embeddings.
    param_iteration: int
        Iteration through paramter list.
    gensim_model_path: str (optional)
        The filepath of the custom gensimModel.
    '''
    import gensim.downloader as api
    import gensim.models as model
    import time
    import os

    st = time.time()

    word_vectors = None
    has_custom_model = gensim_model_path != ""
    if has_custom_model:
        word_vectors = model.Word2Vec.load(gensim_model_path)
    else:
        print('using pretrained model')
        word_vectors = api.load('word2vec-google-news-300')
    missing_words = 0
    iteration = 0
    document_embeddings = []
    for iteration in range(len(pmids)):
        # Retrieve word embeddings.
        embedding_list = []
        if(has_custom_model):
            for word in article_doc[iteration]:
                try:
                    embedding_list.append(word_vectors.wv[word])
                except:
                    missing_words += 1
        else:
            for word in article_doc[iteration]:
                try:
                    embedding_list.append(word_vectors[word])
                except:
                    missing_words += 1

        # Generate document embeddings from word embeddings using word-vector centroids.
        if len(embedding_list) == 0:
            # This can be caused by a high min-count parameter or missing vocabulary when using a pretrained model
            document_embeddings.append([])
            continue
            
        document = [0.0] * word_vectors.vector_size

        for dim in range(word_vectors.vector_size):
            for word_embeddings in embedding_list:
                document[dim] += word_embeddings[dim]
            document[dim] = document[dim] / len(embedding_list)
        document_embeddings.append(document)

    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    
    import pandas as pd
    df = pd.DataFrame(list(zip((pmids), document_embeddings)), columns =['pmids', 'embeddings'])
    df = df.sort_values('pmids')
    os.makedirs(f"{directory_out}/{param_iteration}", exist_ok=True)
    df.to_pickle(f'{directory_out}/{param_iteration}/embeddings.pkl')
    print (f'done for {param_iteration}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
        help="File path to input RELISH tokenized .npy file")
    parser.add_argument("-annoti", "--Annot_input", type=str,
        help="File path to input Annotated RELISH tokenized .npy file")
    parser.add_argument("-o", "--output", type=str,
        help="Path to save embeddings pickle file and the corresponding model")       
    parser.add_argument("-pj", "--params_json", type=str,
        help="File path to the word2vec parameter list.")
    parser.add_argument("-up", "--use_pretrained", type=int,
        help="Whether to use a pretrained model or not") 
    parser.add_argument("-dict", "--MeShIDtoPMID", type=str,
        help="File path to input MeShIDtoPMID .tsv file.")
    parser.add_argument("-rd", "--reduction", type=int,
        help="Whether to reduce the documents'words by replacing the catalogued ones with corresponding MeSHID (1) or not (0)")
    args = parser.parse_args()

    params = []
    with open(args.params_json, "r") as openfile:
        params = json.load(openfile)
    
    model_output_File = ""
    if not args.use_pretrained:
        model_output_File = "./data/word2vec_model"
    
    global_article_Annot_docs_dict = generate_npy_dict(args.Annot_input)
    
    for iteration in range(len(params)):
        print(f'start for {iteration}')
        st = time.time() 
        
        pmids_global, article_doc_global = prepare_from_npy(args.input)
        
        
        generate_Word2Vec_model(article_doc_global, pmids_global, params[iteration], model_output_File, args.use_pretrained)
        injection_MeSHembeddings_into_embeddings(pmids_global, article_doc_global, args.MeShIDtoPMID, 
                                                               args.output, iteration, args.reduction, model_output_File)
        
        generate_document_embeddings(pmids_global, article_doc_global, args.output, iteration, model_output_File)
        
        et = time.time()
        elapsed_time = et - st
        print(f'Execution time for param_set {iteration}: ', elapsed_time, 'seconds')
