# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/train.py
# This file includes the modifications to the source codes according to this project!

import os
import argparse
import utilities as utilities

def run(best_params, args, iteration, save_model=False):
    # Load data
    train_pmids, train_docs = utilities.process_data_from_npy(args.input)
    print("Retrieved RELISH Cleaned Training Data")
    # Train the model
    model = utilities.createWord2VecModel(train_pmids, train_docs, best_params)
    # Finding MeSH-terms in training tokens to compute the corresponding MeSHIDs' embeddings and incorporate them into trained model
    model = utilities.injection_MeSHembeddings_into_embeddings(model, train_pmids, train_docs, args.MeShIDtoPMID)
    print("RELISH Word2Vec Model Generated and MeSHIDs' Embeddings Injected.")
    
    if args.reduction:        
        if args.WMD_score:
            # Store Relish Post-processed/annotated tokens in a dictionary with keys PMIDs
            article_post_annot_docs_dict = utilities.generate_postReduction_npy_dict_via_injection_MeSHIDs_into_tokens(
                args.input, args.MeShIDtoPMID)
            print("Prepared RELISH Post-Annot Dictionary For Hybrid-WMD-PostReduction-Word2Doc2Vec")

            # Define the file path for Storing WMD similarity matrix
            WMD_similarity_file = f"output_of_model/evaluation/WMD_scores/WMD_similarity_{iteration}.tsv"
            # Generate and save the WMD similarity matrix
            utilities.get_and_save_WMD_similarity_scores(args.ground_truth, model, article_post_annot_docs_dict, WMD_similarity_file)
            print("RELISH WMD Similarity Matrix Saved.")
        
        # Replacement of MeSH-terms in tokens with the corresponding MeSHIDs
        train_docs = utilities.replacement_of_MeSHterms_with_MeSHIDs_in_tokens(train_pmids, train_docs, args.MeShIDtoPMID)
        print("START of Generating Document-Embeddings!")
        # Define the file path for Storing Document-Embeddings
        embeddings_file = f"output_of_model/doc_embeddings/embeddings_pickle_{iteration}.pkl"
        # Generate Document-Embeddings
        utilities.generate_document_embeddings(model, train_pmids, train_docs, embeddings_file)
        
        if args.cosine_similarity:
            # Define the file path for Storing cosine similarity matrix
            cosine_similarity_file = f"output_of_model/evaluation/cosine_similarity_{iteration}.tsv"
            # Generate and save the cosine similarity matrix
            utilities.get_and_save_cosine_similarity_scores(args.ground_truth, embeddings_file, cosine_similarity_file)
            print("RELISH Cosine Similarity Matrix Saved")
    
    else:        
        if args.WMD_score:
            # Store Relish Post-processed/annotated tokens in a dictionary with keys PMIDs
            article_post_annot_docs_dict = utilities.generate_post_npy_dict_via_injection_MeSHIDs_into_tokens(
                args.input, args.MeShIDtoPMID)
            print("Prepared RELISH Post-Annot Dictionary For Hybrid-WMD-Post-Word2Doc2Vec")

            # Define the file path for Storing WMD similarity matrix
            WMD_similarity_file = f"output_of_model/evaluation/WMD_scores/WMD_similarity_{iteration}.tsv"
            # Generate and save the WMD similarity matrix
            utilities.get_and_save_WMD_similarity_scores(args.ground_truth, model, article_post_annot_docs_dict, WMD_similarity_file)
            print("RELISH WMD Similarity Matrix Saved.")
        
        # Finding MeSH-terms in data in order to append the corresponding MeSHIDs' into tokens
        train_docs = utilities.injection_MeSHIDs_into_tokens(train_pmids, train_docs, args.MeShIDtoPMID)
        print("START of Generating Document-Embeddings!")
        # Define the file path for Storing Document-Embeddings
        embeddings_file = f"output_of_model/doc_embeddings/embeddings_pickle_{iteration}.pkl"
        # Generate Document-Embeddings
        utilities.generate_document_embeddings(model, train_pmids, train_docs, embeddings_file)
        
        if args.cosine_similarity:
            # Define the file path for Storing cosine similarity matrix
            cosine_similarity_file = f"output_of_model/evaluation/cosine_similarity_{iteration}.tsv"
            # Generate and save the cosine similarity matrix
            utilities.get_and_save_cosine_similarity_scores(args.ground_truth, embeddings_file, cosine_similarity_file)
            print("RELISH Cosine Similarity Matrix Saved")
            
    if save_model:
        # Define the file path for saving the model
        model_file = f"output_of_model/model/{iteration}/Word2Vec_model"
        # Save the model
        utilities.saveWord2VecModel(model, model_file)