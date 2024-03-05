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
        start = time.time()
        # Store annotated validation tokens in a dictionary with keys PMIDs
        global_article_Annot_docs_dict = utilities.generate_npy_dict(args.Annot_valid)
        # use validation dataset for tuning
        test_pmids, test_docs = utilities.process_data_from_npy(args.valid)
        # Finding MeSH-terms in validation data in order to append the corresponding MeSHIDs' to validation tokens
        test_docs = utilities.injection_MeSHIDs_into_tokens(model, test_pmids, test_docs, 
                                                            global_article_Annot_docs_dict, args.MeShIDtoPMID, args.reduction)
        print(f"Retrieved RELISH Cleaned Validation Data with Reduction={args.reduction}")

        # Define a directory for storing embeddings
        embeddings_directory = "output_of_model/doc_embeddings"
        if not os.path.exists(embeddings_directory):
            os.makedirs(embeddings_directory)

        embeddings_file = os.path.join(embeddings_directory, "test_embeddings_pickle.pkl")

        # Generate the Validation embeddings
        utilities.generate_document_embeddings(model, test_pmids, test_docs, embeddings_file)
        print(f"RELISH (Validation) Embeddings Pickle File generated and Saved.")

        # Define the directory for storing evaluation results
        output_directory = "output_of_model/evaluation"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Generate and save the cosine similarity matrix
        similarity_file = os.path.join(output_directory, "cosine_similarity.tsv")
        utilities.get_similarity_scores(args.valid_ground_truth, embeddings_file, similarity_file)
        print("RELISH (Validation) Cosine Similarity Matrix Saved.")
        end = time.time()
        print(f"Time Taken for Validation: {end - start} seconds.")
    
    else:
        # Store annotated test tokens in a dictionary with keys PMIDs
        global_article_Annot_docs_dict = utilities.generate_npy_dict(args.Annot_test)
        # use test dataset for final evaluation
        test_pmids, test_docs = utilities.process_data_from_npy(args.test)
        # Finding MeSH-terms in test data in order to append the corresponding MeSHIDs' to test tokens
        test_docs = utilities.injection_MeSHIDs_into_tokens(model, test_pmids, test_docs, 
                                                            global_article_Annot_docs_dict, args.MeShIDtoPMID, args.reduction)
        print(f"Retrieved RELISH Cleaned Test Data with Reduction={args.reduction}")

        # Define a directory for storing embeddings
        embeddings_directory = "output_of_model/doc_embeddings"
        if not os.path.exists(embeddings_directory):
            os.makedirs(embeddings_directory)

        embeddings_file = os.path.join(embeddings_directory, "test_embeddings_pickle.pkl")

        # Generate the embeddings
        utilities.generate_document_embeddings(model, test_pmids, test_docs, embeddings_file)
        print("RELISH (Test) Embeddings Pickle File Saved")

        # Define the directory for storing evaluation results
        output_directory = "output_of_model/evaluation"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Generate and save the cosine similarity matrix
        similarity_file = os.path.join(output_directory, "cosine_similarity.tsv")
        utilities.get_similarity_scores(args.test_ground_truth, embeddings_file, similarity_file)
        print("RELISH (Test) Cosine Similarity Matrix Saved")
        
    if save_model:
        # Define the directory for saving the model
        model_directory = "output_of_model/model"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        
        model_file = os.path.join(model_directory, "best_Word2Vec_model")
        # Save the model
        utilities.saveWord2VecModel(model, model_file)
        #utilities.saveWord2VecModel(model, "output_of_model/model/best_Word2Vec_model")



    return similarity_file