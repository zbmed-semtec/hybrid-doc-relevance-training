# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/train.py
# This file includes the modifications to the source codes according to this project!

import os
import time
import argparse
import utilities as utilities

def run(best_params, args, tuning=False, save_model=False):
    
    # Load the training data
    train_pmids, train_docs = utilities.process_data_from_npy(args.input)
    print("Retrieved RELISH Cleaned Training Data")

    start = time.time()
    # Train the model with 80% of the data (i.e. training data) and best parameters
    model = utilities.createWord2VecModel(train_pmids, train_docs, best_params)
    # Finding MeSH-terms in training tokens to compute the corresponding MeSHIDs' embeddings and incorporate them into the generated model
    model = utilities.injection_MeSHembeddings_into_embeddings(model, train_pmids, train_docs, args.MeShIDtoPMID)
    end = time.time()
    print(f"Time taken to train the model: {end - start} seconds.")
    print("RELISH Word2Vec Model Generated and MeSHIDs' Embeddings Injected.")
    print(model, "Model is being used.")
    
    # Load the validation data
    if tuning:
        start = time.time()        
        
        # Store Validation Relish Post-processed/annotated tokens in a dictionary with keys PMIDs
        article_post_annot_docs_dict = utilities.generate_post_npy_dict_via_injection_MeSHIDs_into_tokens(args.valid, args.MeShIDtoPMID)
        print("Prepared RELISH Post-Annot Validation Dictionary For Hybrid-WMD-Post-Word2Doc2Vec")
        # Here similarity_file is a pd.DataFrame
        similarity_file = utilities.get_WMD_similarity_scores(args.valid_ground_truth, model, article_post_annot_docs_dict)
        print("RELISH (Validation) WMD-Similarity-Matrix DataFrame Generated.")
        end = time.time()
        print(f"Time Taken for Validation: {end - start} seconds.")
    
    # Load the test data
    else:   
        start = time.time()
        # Store Test Relish Post-processed/annotated tokens in a dictionary with keys PMIDs
        article_post_annot_docs_dict = utilities.generate_post_npy_dict_via_injection_MeSHIDs_into_tokens(args.test, args.MeShIDtoPMID)
        print("Prepared RELISH Post-Annot Test Dictionary For Hybrid-WMD-Post-Word2Doc2Vec")

        # Define the directory for storing evaluation results
        output_directory = "output_of_model/evaluation"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Generate and save the WMD similarity matrix
        similarity_file = os.path.join(output_directory, "WMD_similarity.tsv")
        utilities.get_and_save_WMD_similarity_scores(args.test_ground_truth, model, article_post_annot_docs_dict, similarity_file)
        print("RELISH (Test) WMD Similarity Matrix Saved ... and Generating Test-embeddings START!")
        
        # Retrieve data from Test RELISH dataset for generating embeddings
        test_pmids, test_docs = utilities.process_data_from_npy(args.test)
        # Find MeSH-terms in Test data in order to append the corresponding MeSHIDs' to Test tokens
        test_docs = utilities.injection_MeSHIDs_into_tokens(model, test_pmids, test_docs, args.MeShIDtoPMID)
    
        # Define the file path for Storing Embeddings of the test Set
        embeddings_file = "output_of_model/doc_embeddings/test_embeddings_pickle.pkl"
        # Generate Embeddings for the Best Validation Hyperparameter Set
        utilities.generate_document_embeddings(model, test_pmids, test_docs, embeddings_file)
        
        end = time.time()
        print(f"Time Taken for Test-Phase: {end - start} seconds.")
    
    if save_model:
        # Define the file path for saving the model
        model_file = "output_of_model/model/Word2Vec_model"
        # Save the model
        utilities.saveWord2VecModel(model, model_file)

    
    return similarity_file, model
