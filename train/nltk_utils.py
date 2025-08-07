from sentence_transformers import SentenceTransformer

# Load the pretrained model once
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed_sentence(sentence):
    """
    Generate a 384-dim embedding from a sentence using SentenceTransformer.
    Returns a NumPy array.
    """
    return embedder.encode([sentence])[0]