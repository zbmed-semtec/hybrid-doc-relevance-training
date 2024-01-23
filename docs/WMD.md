# Word Mover’s Distance

The Word Mover's Distance (WMD) quantifies the **dissimilarity** between two text documents by determining the minimum distance necessary for the embedded words of one document to "travel" and align with the embedded words of another document [1](http://mkusner.github.io/publications/WMD.pdf)

##How to calculate WMD

For a finite size vocabulary of $n$ words, assume that the word2vec embeddings are generated for two text documnets $x$ and $y$ such that e.g. $emb_{x_i}$ denotes the embeddings of document $x$'s $i^\text{th}$ word. Then, the transportation cost for travelling between document $x$'s $i^\text{th}$ word and document $y$'s $j^\text{th}$ word can be defined as the distance between the corresponding embeddings, i.e. 
$$C_{ij} = distance(emb_{x_i},emb_{y_j}).$$
Note that the normalized frequency of word $i$, which appears \nu_i times in a document, is defined as $f_i = \frac{\nu_i}{\sum_{k=1}^n \nu_k}$.
WMD score between two text documnets $x$ and $y$ is calculated as
$$WMD(x,y) = \min_{F\ge 0}\sum_{i,j=1}^n F_{ij} C_{ij},$$
subject to 
$$\sum_{i=1}^n F_{ij} = f_x_i, \text{for all }i\in{1,...,n},$$
$$\sum_{j=1}^n F_{ij} = f_y_j, \text{for all }j\in{1,...,n}.$$
Here $F_{ij}$ represents how much of document $x$'s $i^\text{th}$ word travels to document $y$'s $j^\text{th}$ word.
Here, $F_{ij}$ denotes the extent to which the $i^\text{th}$ word of document $x$ transitions to the $j^\text{th}$ word of document $y$

