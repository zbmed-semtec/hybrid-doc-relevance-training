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
    
    # Load the validation/test data
    if tuning:
        # use validation dataset for tuning
        start = time.time()
        if args.reduction:
            # Tokens of postreduction case are the same as pre-annotated tokens:
            test_pmids, test_docs = utilities.process_data_from_npy(args.Annot_valid)
        else:
            test_pmids, test_docs = utilities.process_data_from_npy(args.valid)
            # Finding MeSH-terms in validation data in order to append the corresponding MeSHIDs' to validation tokens
            test_docs = utilities.injection_MeSHIDs_into_tokens(test_pmids, test_docs, args.MeShIDtoPMID)
        
        print(f"Retrieved RELISH Cleaned Validation Data with Reduction={args.reduction}")

        # Generate the Validation embeddings: Here embeddings_file is a pd.DataFrame
        embeddings_file = utilities.generate_document_embeddings(model, test_pmids, test_docs)
        print(f"RELISH (Validation) Embeddings generated.")
        # Here similarity_file is a pd.DataFrame
        similarity_file = utilities.get_similarity_scores(args.valid_ground_truth, embeddings_file)
        print("RELISH (Validation) Cosine Similarity Matrix Generated.")
        end = time.time()
        print(f"Time Taken for Validation: {end - start} seconds.")
    
    else:
        # use test dataset for final evaluation
        start = time.time()
        if args.reduction:
            # Tokens of postreduction case are the same as pre-annotated tokens:
            test_pmids, test_docs = utilities.process_data_from_npy(args.Annot_test)
        else:
            test_pmids, test_docs = utilities.process_data_from_npy(args.test)
            # Finding MeSH-terms in test data in order to append the corresponding MeSHIDs' to test tokens
            test_docs = utilities.injection_MeSHIDs_into_tokens(test_pmids, test_docs, args.MeShIDtoPMID)
        
        print(f"Retrieved RELISH Cleaned Test Data with Reduction={args.reduction}")
        
        # Define the file path for Storing test Embeddings
        embeddings_file = "output_of_model/doc_embeddings/test_embeddings_pickle.pkl"

        # Generate the embeddings
        embeddings_df = utilities.generate_document_embeddings(model, test_pmids, test_docs)
        # Store the embeddings
        utilities.save_embeddings_to_pickle(embeddings_df, embeddings_file)
        print("RELISH (Test) Embeddings Pickle File Saved")

        # Define the file path for Storing cosine similarity matrix
        similarity_file = "output_of_model/evaluation/cosine_similarity.tsv"
        # Generate and save the cosine similarity matrix
        utilities.get_and_save_similarity_scores(args.test_ground_truth, embeddings_file, similarity_file)
        print("RELISH (Test) Cosine Similarity Matrix Saved")
        end = time.time()
        print(f"Time Taken for Test-Phase: {end - start} seconds.")
        
       
    if save_model:
        # Define the file path for saving the model        
        model_file = "output_of_model/model/Word2Vec_model"
        # Save the model
        utilities.saveWord2VecModel(model, model_file)
    

    return similarity_file, embeddings_file, model