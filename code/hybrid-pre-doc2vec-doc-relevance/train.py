# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/train.py
# This file includes the modifications to the source codes according to this project!

import os
import time
from gensim.models import Doc2Vec
import argparse
import utilities as utilities

def run(best_params, args, tuning=False):
    
    # Validation Phase
    if tuning:
        # Load the training data
        train_pmids, train_docs = utilities.process_data_from_npy(args.input)
        print("Retrieved RELISH Cleaned Training Data")

        start = time.time()
        # Train the model with 80% of the data (i.e. training data) and best parameters
        model = utilities.createDoc2VecModel(train_pmids, train_docs, best_params)
        print("RELISH Hybrid Dord2Vec Model Generated.")
        print(model, "Model is being used.")
    
        # use validation dataset for tuning
        test_pmids, test_docs = utilities.process_data_from_npy(args.valid)
        print(f"Retrieved RELISH Cleaned Validation Data")

        # Generate the Validation embeddings: Here embeddings_file is a pd.DataFrame
        embeddings_file = utilities.generate_embeddings(model, test_pmids, test_docs)
        print(f"RELISH (Validation) Embeddings Pickle File Generated.")

        # Generate cosine similarity matrix: Here similarity_file is a pd.DataFrame
        similarity_file = utilities.get_similarity_scores(args.valid_ground_truth, embeddings_file)
        print("RELISH (Validation) Cosine Similarity Matrix Generated.")
        end = time.time()
        print(f"Time Taken for Validation: {end - start} seconds.")
        
        return similarity_file, embeddings_file, model
    
    else: # Test Phase
        
        start = time.time()
        
        # Load the previously saved best-trained model from the validation phase
        model = Doc2Vec.load("output_of_model/model/best_Doc2Vec_model")
 
        # use test dataset for final evaluation
        test_pmids, test_docs = utilities.process_data_from_npy(args.test)
        print(f"Retrieved RELISH Cleaned Test Data")

        # Define the file path for Storing test Embeddings
        embeddings_file = f"output_of_model/doc_embeddings/test_embeddings_pickle_{args.classes}.pkl"
        # Generate the embeddings
        embeddings_df = utilities.generate_embeddings(model, test_pmids, test_docs)
        # Store the embeddings
        utilities.save_embeddings_to_pickle(embeddings_df, embeddings_file)
        print("RELISH (Test) Embeddings Pickle File Saved")

        # Define the file path for Storing cosine similarity matrix
        similarity_file = f"output_of_model/evaluation/cosine_similarity_{args.classes}.tsv"
        # Generate and save the cosine similarity matrix
        utilities.get_and_save_similarity_scores(args.test_ground_truth, embeddings_file, similarity_file)
        print("RELISH (Test) Cosine Similarity Matrix Saved")
        
        end = time.time()
        print(f"Time Taken for Test-Phase: {end - start} seconds.")


        return similarity_file