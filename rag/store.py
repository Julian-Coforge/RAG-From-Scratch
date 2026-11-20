from sentence_transformers import SentenceTransformer


def make_embedder(model_name):
    """
    Creates an embedding function using the specified model from the sentence-transformers library.
    """
    # Load the specified model from the sentence-transformers library
    model = SentenceTransformer(model_name)

    def embed_function(texts):
        """
        Generates embeddings for a list of texts using the loaded model.
        """
        return model.encode(texts, convert_to_numpy=True).tolist()

    return embed_function

def add_chunks(collection, chunks, embed_function):
    """
    Adds a list of chunks to a ChromaDB collection, embedding each chunk using the provided embedding function.
    """

    if not chunks:
        return

    # Extract the text of each chunk to be embedded
    texts = [chunk["chunk"] for chunk in chunks]
    
    # Generate embeddings for the chunks using the provided embedding function
    embeddings = embed_function(texts)
    
    # Prepare the data to be added to the collection
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [{k: v for k, v in chunk.items() if k not in ("id", "chunk")} for chunk in chunks]

    # Add the chunks to the collection with their embeddings and metadata
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

def search(collection, query, embed_function, k=5):
    """
    Searches for the top-k most relevant chunks in a ChromaDB collection based on a query.
    """ 

    # Generate the embedding for the query using the provided embedding function
    query_embedding = embed_function([query])[0]
    
    # Perform the search in the collection using the query embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    # Extract relevant information from the search results
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    
    return [
        {
            **metadata,
            "id": id_,
            "chunk": doc,
            "distance": float(distance),
        }
        for id_, doc, metadata, distance in zip(ids, documents, metadatas, distances)
    ]

