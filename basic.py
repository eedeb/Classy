# import packages
import random
import torch
from Classy.model import NeuralNet
import nltk
from nltk.stem.porter import PorterStemmer
# define variables
stemmer = PorterStemmer()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data=0
input_size = ''
hidden_size = ''
output_size = ''
tags = ''
model_state = ''

model = 'a'
check=False
# get info from init
def init(location):
    global data
    global input_size
    global hidden_size
    global output_size
    global tags
    global model_state
    global model
    global check
    data = torch.load(location)
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    check=True
# define funcions for processing output
import numpy as np

from sentence_transformers import SentenceTransformer

# Load the pretrained model once
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed_sentence(sentence):
    """
    Generate a 384-dim embedding from a sentence using SentenceTransformer.
    Returns a NumPy array.
    """
    return embedder.encode([sentence])[0]
# function for classifying input
def classify(sentence,location):
    ignore_words = ['?', '.', '!', '"']
    sentence = sentence.lower()
    for i in ignore_words:
        sentence = sentence.replace(i, '')
    # make sure the model has been initialized
    global check
    if check == False:
        init(location)
    # Generate embedding
    X = embed_sentence(sentence)  # This returns a 384-dim vector (np array)
    X = torch.from_numpy(X).float().unsqueeze(0).to(device)  # shape: [1, 384]
    # get output from the model
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob=[]
    for i in probs[0]:
        prob.append(i.item())
    old_prob=prob.copy()
    prob.sort(reverse=True)
    likely_tags=[]
    for i in prob:
        prob_index=old_prob.index(i)
        likely_tags.append(tags[prob_index])
    #prob = probs[0][predicted.item()]
    return likely_tags, prob
