This code generates a tsv-file s.t. the first column is detected MeSHIDs in the annotated Relish xml-files and the second column is a list of pairs (PMID , word). Here PMID corresponds to the article containing the MeSHID and word/term is the tagged word/term by that MeSHID in the article. 
Tagged words/terms undergo some kind of preprocessing, such as converting to lowercase and tokenization before storing in the tsv-file.
To execute the script and generate a single TSV file, you can run one of the following commands:

    python3 code/xml_translate/generate_Dic_MeShIDtoPMID.py [-d INPUT DIRECTORY PATH] [-o OUTPUT PATH]
    
    or
    
    python3 code/xml_translate/generate_Dic_MeShIDtoPMID.py [-i INPUT PATH] [-o OUTPUT PATH]

You must pass the following arguments:

    -i/ --input : Path to input XML file.
    -d/ --indir : Path to input directory of XML files.
    -o/ --output : Path to output tsv file.
    
For example to run this script, you may execute the following command:

    python3 code/xml_translate/generate_Dic_MeShIDtoPMID.py --indir data/sample_annotated_xml --output data/dic_MeShIDtoPMID.tsv
