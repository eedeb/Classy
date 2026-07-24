from sentence_transformers import SentenceTransformer

# Load the pretrained model once
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed_sentence(sentence):
    """
    Generate a 384-dim embedding from a sentence using SentenceTransformer.
    Returns a NumPy array.
    """
    return embedder.encode([sentence])[0]


def embed_sentences(sentences):
    """
    Embed a list of sentences in a single batched call. Much faster than
    calling embed_sentence() in a loop. Returns a (N, 384) NumPy array.
    """
    return embedder.encode(list(sentences), batch_size=64, show_progress_bar=True)
