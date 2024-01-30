'''
Example
-------
To execute the script and generate document embeddings, you can run the following command:

python3 code/meshd_to_MeSHD.py --input data/RELISH/Tokenized_Input/RELISH_Annot_Tokens_Sample.npy --output data/RELISH/Tokenized_Input/RELISH_Annot_Tokens_Sample_meshd_to_MeSHD.npy

'''
import argparse
import numpy as np
import re

def prepare_from_npy(filepath_in: str, filepath_out: str):
    '''
    Change annotated-terms' prefix from meshd/meshq to MeSHD/MeSHQ in the pre-annotated tokenized npy file: required for WMD due to generated word2vec 
    model w.r.t. MeSHD/MeSHQ, specifically embeddings in the tarined model are labelled by prefix MeSHD/MeSHQ.

    Parameters
    ----------
    filepath_in: str
        The filepath of the RELISH input npy file.
    filepath_out: str
        The filepath of the RELISH output npy file.
    '''
    doc = np.load(filepath_in, allow_pickle=True)
    for line in doc:
        line[1] = [re.sub("meshd", 'MeSHD', str(word)) for word in line[1]]
        line[1] = [re.sub("meshq", 'MeSHQ', str(word)) for word in line[1]]
        
        line[2] = [re.sub("meshd", 'MeSHD', str(word)) for word in line[2]]
        line[2] = [re.sub("meshq", 'MeSHQ', str(word)) for word in line[2]]
        
    np.save(filepath_out, doc, allow_pickle=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type=str,
                       help="Path to input tokenized NPY file")
    parser.add_argument("-o", "--output", type=str,
                       help="Path to output tokenized NPY file")
    args = parser.parse_args()

    prepare_from_npy(args.input, args.output)
