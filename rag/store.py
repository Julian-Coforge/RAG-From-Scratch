from sentence_transformers import SentenceTransformer


def make_embedder(model_name):
    """
    Creates an embedding function using the specified model from the sentence-transformers library.

    Args:
        model_name: The name of the model to use for generating embeddings.
    Returns:
        A function that takes a list of texts and returns their embeddings.
    """
    # Load the specified model from the sentence-transformers library
    model = SentenceTransformer(model_name)

    def embed_function(texts):
        """
        Generates embeddings for a list of texts using the loaded model.

        Args:
            texts: A list of strings to be embedded.
        Returns:
            A list of embeddings corresponding to the input texts.
        """
        return model.encode(texts, convert_to_numpy=True).tolist()

    return embed_function

def add_chunks(collection, chunks, embed_function):
    """
    Adds a list of chunks to a ChromaDB collection, embedding each chunk using the provided embedding function.

    Args:
        collection: The ChromaDB collection to which the chunks will be added.
        chunks: A list of dictionaries, each containing 'id', 'chunk', 'section', and 'source' keys.
        embed_function: A function that takes a list of texts and returns their embeddings.
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

    Args:
        collection: The ChromaDB collection to search.
        query: The search query string.
        embed_function: A function that takes a list of texts and returns their embeddings.
        k: The number of top results to return (default is 5).
    Returns:
        A list of dictionaries, each containing 'id', 'chunk', 'section', and 'source' keys for the top-k most relevant chunks.
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
    distance = results["distances"][0]
    
    # Combine the extracted information into a list of dictionaries for each result
    return [
        {
            "id": id_,
            "chunk": doc,
            "section": metadata.get("section"),
            "source": metadata.get("source"),
            "distance": distance
        }
        for id_, doc, metadata, distance in zip(ids, documents, metadatas, distance)
    ]
