import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    
    docs = vec_docs.toarray()
    queries = vec_queries.toarray()
    n_iter = 2
    alpha = 0.7
    beta = 0.3

    for i in range(n_iter):
        # print(i)
        retrieved_idx = np.argsort(-sim, axis=0)
        
        retrieved_docs = []
        for row in retrieved_idx:
            retrieved_docs.append([docs[idx] for idx in row])
        retrieved_docs = np.array(retrieved_docs)
        
        rel = retrieved_docs[:n]
        non_rel = retrieved_docs[-n:]
        
        sum_rel = np.sum(rel, axis=0)
        sum_non_rel = np.sum(non_rel, axis=0)
        
        queries = queries + (alpha * sum_rel) - (beta * sum_non_rel)
        sim = cosine_similarity(docs, queries)

    rf_sim = sim
    return rf_sim

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def arrtostr(ary):
    string = ''
    for x in ary:
        string += str(x)+ ' '
    return string

def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    
    docs = vec_docs.toarray()
    queries = vec_queries.toarray()
    n_iter = 2
    alpha = 0.7
    beta = 0.3
    words = tfidf_model.get_feature_names()
    k = 20 # number of terms selected

    for i in range(n_iter):
        # print(i)
        retrieved_idx = np.argsort(-sim, axis=0)
        
        retrieved_docs = []
        for row in retrieved_idx:
            retrieved_docs.append([docs[idx] for idx in row])
        retrieved_docs = np.array(retrieved_docs)
        
        rel = retrieved_docs[:n]
        non_rel = retrieved_docs[-n:]
        
        sum_rel = np.sum(rel, axis=0)
        sum_non_rel = np.sum(non_rel, axis=0)
        
        queries = queries + (alpha * sum_rel) - (beta * sum_non_rel)
        
        extensions = [] 
        for q in range(len(queries)):
            query_i = [q] * k
            # Top k terms calculated for each query (over all relevant docs)
            doc_i, term_i = largest_indices(rel[:,q], k)
            # Top k terms appended to form query extension
            extension_words = [words[i] for i in term_i]
            extensions.append(arrtostr(extension_words))    
        # Calculated tf-idf vector of each query extension
        extensions = tfidf_model.transform(extensions).toarray()    

        queries = queries + extensions
        
        sim = cosine_similarity(docs, queries)

    rf_sim = sim  # change
    return rf_sim