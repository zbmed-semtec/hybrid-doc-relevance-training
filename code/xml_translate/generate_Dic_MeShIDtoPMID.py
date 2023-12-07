"""
***This code is derived from xml_translate.py (coded by "Guillermo Rocamora Pérez") in hybrid_doc2vec model***

This file aims to generate a tsv file in form [MeSHID, [(PMID , tokenized tagged-terms)]] from the annotated XML files obtained from
[Whatizit](https://github.com/zbmed-semtec/whatizit-dictionary-ner).

Example
-------
To execute the script and generate a single TSV file, you can run the following
command:

    python3 code/xml_translate/generate_Dic_MeShIDtoPMID.py --indir data/sample_annotated_xml --output data/dic_MeShIDtoPMID.tsv
    
""" 

import os
import sys

import argparse
import glob
import re
import logging
import pandas as pd

from nltk.tokenize import word_tokenize

from xml.etree import ElementTree as ET
from typing import Union, List


logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def argument_parser(args: argparse.ArgumentParser) -> Union[List[str], List[str]]:
    """
    Reads the arguments and creates the necessary variables for the program to
    operate properly.

    Parameters
    ----------
    args: argparse.ArgumentParser
        Arguments from argparse.

    Returns
    -------
    files_in: list[str]
        List of input files.
    """
    if args.indir:
        if args.indir.endswith(".xml"):
            logger.error(
                f"Your input is not a directory, please use --input instead.", exc_info=False)
            sys.exit("No valid input directory.")

        indir = args.indir.rstrip("/")
        files_in = glob.glob(indir + "/*.xml")
        

        if not files_in:
            logger.error(
                f"No XML files located in the input directory.", exc_info=False)
            sys.exit("No valid input directory.")

    elif args.input:
        files_in = [args.input]

    return files_in


class XMLtrans:
    """
    Class to handle every aspect of the translation algorithm. Each object
    should correspond to a given XML file to be modified.

    Attributes
    ----------
    root: xml.etree.ElementTree.Element
        Root object of xml.etree after parsing the document.
    mesh_id_pattern: str
        Regular expression used to match the MeSH ID.
    namespace: dict
        Dictionary containing the namespace for the XML.
    """

    def __init__(self, input_file: str, verify_integrity: bool = False) -> None:
        """
        Initialize an object from XMLtrans class.

        Parameters
        ----------
        input_file : str
            Path for input file.
        verify_integrity : bool, optional
            Whether to verify that all tagged concepts share the same MeSH ID,
            by default False. Behaviour not properly tested. Use at your own
            risk.
        """
        try:
            self.tree = ET.parse(input_file)
            self.root = self.tree.getroot()
        except Exception:
            logger.error(
                f"Input file ({input_file}) is not a valid XML file.", exc_info=True)
            sys.exit("Input file must have a valid XML format.")

        self.mesh_id_pattern = r"\/MESH\/(.*)"
        
        # Important: modify the namespace if necessary. The program will not
        # recognize any annotation or MeSH ID if the namespace is not correct.
        self.namespace = {
            "z": "https://github.com/zbmed-semtec/whatizit-dictionary-ner#"}

        self.pmid = self.locate_pmid()

    def locate_pmid(self) -> str:
        """
        Locates the ID tag inside the XML file and returns it.

        Returns
        -------
        str
            PMID of the publication.
        """
        if self.root.find("document/id") is None:
            logger.warning("No PMID was found in the document. Defaults to 0")
            return 0

        return self.root.find("document/id").text.strip()

    def extract_mesh_id(self, tag: ET.Element) -> str:
        """
        From a matched z:mesh tag, it extracts the correspondent MeSH ID if the
        field "id" is found.

        Parameters
        ----------
        tag : ET.Element
            Object correspondent of a <z:mesh></z:mesh> tag.

        Returns
        -------
        mesh_id: str
            Text with the mesh ID.
        """
        if tag.attrib.get("id"):
            mesh_id = "MeSH" + \
                re.search(self.mesh_id_pattern, tag.attrib.get("id")).group(1)
        else:
            mesh_id = tag.text.strip()
        return mesh_id

    def create_dict(self, MeShIDtoPMID_dic: dict = {}) -> dict:
        """
        Creates the detection dictionary.

        Parameters
        ----------
        MeShIDtoPMID_dic: dict
            The detection dictionary that matches the detected MeShIDs to their corresponding
            terms and PMIDs where corresponding terms appear in (their titles or abstracts).
            Here PMIDs correspond to the already processed articles!

        Returns
        -------
        MeShIDtoPMID_dic: dict
            The PMID and its tokenized lowercase tagged-terms insert in the corresponding MeSHID.
        """

        for tagged in self.root.findall("document/passage/text/z:mesh", self.namespace):
            mesh_id = str(self.extract_mesh_id(tagged))
            pmid_word = [int(self.pmid)] + word_tokenize(tagged.text.lower())
            
            if MeShIDtoPMID_dic.get(mesh_id, []) and pmid_word not in MeShIDtoPMID_dic[mesh_id]:
                MeShIDtoPMID_dic[mesh_id].append(pmid_word)
            else:
                MeShIDtoPMID_dic[mesh_id] = [pmid_word] 
                
        return MeShIDtoPMID_dic


def translate_pipeline(files_in: List[str], output_file: str) -> pd.DataFrame:
    """
    Pipeline to translate all input files and to output a TSV file.

    Parameters
    ----------
    files_in : List[str]
        List of input files.
    output_file : str
        Path for output file.

    Returns
    -------
    MeShIDtoPMID_df: pd.DataFrame
        Generated pandas dataframe containing three columns: "MeSHID" and "Appearance(pmid , tokenized lowercase words)"
    """

    if not output_file.endswith(".tsv"):
        output_file = output_file + ".tsv"
        
    MeShIDtoPMID_dic = {}
    
    for _, file in enumerate(files_in):
        xml_translation = XMLtrans(file)
        xml_translation.create_dict(MeShIDtoPMID_dic)
        
    mesh_pmid_word = []  
    for key, value in MeShIDtoPMID_dic.items():
        mesh_pmid_word.append((key, value))
        
    MeShIDtoPMID_df = pd.DataFrame(mesh_pmid_word, columns = ['MeSHID', 'Appearance(pmid , tokenized lowercase words)'])
    MeShIDtoPMID_df.to_csv(output_file, sep="\t", index=False, quotechar="`")

    return MeShIDtoPMID_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", type=str,
                       help="Path to input XML file")
    group.add_argument("-d", "--indir", type=str,
                       help="Path to input folder with XML files")
    parser.add_argument("-o", "--output", type=str,
                        help="Path to output text file")
    
    args = parser.parse_args()

    files_in = argument_parser(args)
    translate_pipeline(files_in, args.output)
