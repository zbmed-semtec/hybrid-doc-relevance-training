# Word Mover’s Distance

The Word Mover's Distance (WMD) quantifies the **dissimilarity** between two text documents by determining the minimum distance necessary for the embedded words of one document to "travel" in the word embedding space and align with the embedded words of another document [see [1](http://mkusner.github.io/publications/WMD.pdf)].

## Definition

Word Mover's Distance (WMD) is a method for measuring the dissimilarity between two text documents, taking into account the semantic meaning of words. It was introduced by Matt Kusner, Yu Sun, Nicholas Colwell, Weixuan Fu, and Kilian Q. Weinberger in their 2015 paper "[From Word Embeddings To Document Distances](http://mkusner.github.io/publications/WMD.pdf)".

Here's a detailed explanation of how WMD works:

1. **Word Embeddings:**
   - WMD relies on the use of word embeddings, which are vector representations of words in a continuous vector space. Word embeddings capture semantic relationships between words, and similar words are represented as vectors that are close to each other in this space. That is to say word embeddings are trained to represent semantic similarity, meaning that words with similar meanings have similar vector representations.
   - Popular methods for generating word embeddings include Word2Vec, GloVe, and FastText: Here we use Word2Vec method.

2. **Document Representation:**
   - Each document is represented as a bag-of-words, where each word in the document is associated with its word embedding vector.
   - The bag-of-words representation maintains the count of each word in the document, but it loses information about word order. Specifically, in a typical bag-of-words representation, the order of words is disregarded, and only the frequency or count of each word is considered. This representation is a simple and common way to represent text, but it neglects the sequential information.

3. **Word Movement:**
   - WMD calculates the distance between two documents by measuring the minimum "cost" required to move the embedded words from one document to match the distribution of words in the other document. To be more precise, WMD evaluates the distance between two documents by finding the optimal transport plan that minimizes the cost of moving the "mass" (words) from one document to another in the embedding space.
   - The cost is usually defined as the Euclidean distance between word vectors. This is due to the fact that the distance between words in the embedding space reflects both semantic similarity and the potential "cost" of moving from one word to another

4. **Optimization:**
   - The optimization problem involves finding the optimal transport plan, i.e., the most efficient way to transform the word distribution of one document into that of another. Specifically it involves finding the most efficient way to "move" words from one document to another in the embedding space.
   - The transportation cost is the sum of the products of the distance between words and the amount of "mass" transported.
   - If the order of words changes in one document compared to another, it will affect the optimal transport plan because the algorithm has to account for the distance between the corresponding words.

5. **Calculating WMD:**
   - The WMD between two documents is computed by finding the optimal transport plan that minimizes the total cost of moving words from one document to another.
   - The resulting distance reflects the semantic **dissimilarity** between the two documents, considering both the meaning of the words and their distribution.

6. **Complexity:**
   - Computing the WMD can be computationally expensive, especially when dealing with large vocabularies. Various optimization techniques and approximations are often employed to make the calculation more feasible.

In summary, Word Mover's Distance is a measure of dissimilarity that considers both the meaning of words and their distribution in documents. To be more precise, even though the bag-of-words representation loses explicit information about word order, the use of word embeddings in WMD allows the algorithm to consider both semantic similarity and the "cost" of moving words in the embedding space. It provides a more nuanced understanding of document dissimilarity compared to traditional distance measures like cosine similarity or Jaccard similarity, as it accounts for the semantic relationships between words.

## How to calculate WMD

The transportation plan $T$ in the context of Word Mover's Distance (WMD) represents the optimal way to move the "mass" from the words in one document to the words in another, minimizing the overall cost. This plan is essentially a matrix where each entry $T_{ij}$ indicates the amount of mass to be moved from the $i$-th word in the first document to the $j$-th word in the second document.

Let's go first into more precise details about the transportation plan $T$:

1. **Definition of $T_{ij}$:**
   - $T_{ij}$ represents the amount of mass (words) from the $i$-th word in the first document that is moved to the $j$-th word in the second document.

2. **Constraints on $T$:**
   - The transportation plan $T$ must satisfy constraints to ensure that the mass is conserved. The constraints are defined by the counts of words in each document:
      - $\sum_j T_{ij} = \text{count}(w_i^{(1)})$: The sum of mass from the $i$-th word in the first document must equal the count of that word in the first document.
      - $\sum_i T_{ij} = \text{count}(w_j^{(2)})$: The sum of mass at the $j$-th word in the second document must equal the count of that word in the second document.
      - $T_{ij} \geq 0$: The amount of mass moved from $i$-th word to $j$-th word must be non-negative.

3. **Objective Function:**
   - The objective function of the optimization problem is to minimize the overall cost, which is the sum of the products of the transportation amounts and the distances between the corresponding word vectors:
      $$\min_{T} \sum_{i=1}^{n} \sum_{j=1}^{m} T_{ij} \cdot d(w_i^{(1)}, w_j^{(2)})$$

4. **Solving the Optimization Problem:**
   - Linear programming techniques are often used to solve this optimization problem efficiently and find the optimal transportation plan $T$.

5. **Interpretation:**
   - Once the optimization is complete, the transportation plan $T$ provides a blueprint for how much mass (words) needs to be moved from each word in the first document to align with the word distribution in the second document.

In summary, the transportation plan $T$ is a matrix that specifies the optimal way to move words from one document to another, minimizing the overall cost while satisfying conservation constraints. It is a key component of the Word Mover's Distance algorithm.

Let $D1$ and $D2$ be two documents represented as bags-of-words, where $D1 = (w_{1}^{(1)}, w_{2}^{(1)}, \ldots, w_{n}^{(1)})$ and $D2 = (w_{1}^{(2)}, w_{2}^{(2)}, \ldots, w_{m}^{(2)})$.

Let $X = (x_1, x_2, \ldots, x_n)$ and $Y = (y_1, y_2, \ldots, y_m)$ be the embedded word vectors corresponding to the words in $D1$ and $D2$, respectively.

The distance between two words $w_i^{(1)}$ and $w_j^{(2)}$ is given by the Euclidean distance between their corresponding word vectors:

$$d(w_i^{(1)}, w_j^{(2)}) = \|x_i - y_j\|$$

The WMD between $D1$ and $D2$ is the minimum "cost" required to move the mass from $D1$ to $D2$, and it is defined via the optimization problem as follows:

$$WMD(D1, D2) = \min_{T} \sum_{i=1}^{n} \sum_{j=1}^{m} T_{ij} \cdot d(w_i^{(1)}, w_j^{(2)}),$$

subject to 

$$\sum_{j=1}^{m} T_{ij} = \text{count}(w_i^{(1)}) ,\quad \text{for all }i\text{  s.t.  } 1\leq i \leq n,$$

$$\sum_{i=1}^{n} T_{ij} = \text{count}(w_j^{(2)}) ,\quad \text{for all }j\text{  s.t.  } 1\leq j \leq m, $$

$$T_{ij} \geq 0. $$

In practice, linear programming techniques are often used to solve this optimization problem efficiently. The resulting WMD provides a measure of dissimilarity that considers both the semantic meaning of words and their distribution in the documents.

## Crucial points

The order of words does affect Word Mover's Distance (WMD). Word Mover's Distance is a measure of the dissimilarity between two text documents, taking into account the semantic similarity of words and their respective distances in the word embedding space. In WMD, the order of words is crucial because it considers the relationship and proximity of words in the documents.

## Implemenation of the WMD

Here, we use [Gensim’s implemenation of the WMD](https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html), which uses [word2vec](https://rare-technologies.com/word2vec-tutorial/) vector embeddings of words. Since WMD is the measure of dissimilarity, we define $(1 + WMD)^{-1}$ as similarity score.
