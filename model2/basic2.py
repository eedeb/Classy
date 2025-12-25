from sentence_transformers import SentenceTransformer
import json
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
def relu(x):
    return max(0, x)
def relu_derivative(x):
    return 1.0 if x > 0 else 0.0
def cross_entropy(y_true, y_pred):
    # small epsilon to avoid log(0)
    eps = 1e-8
    return -sum(y_true[i] * np.log(y_pred[i] + eps) for i in range(len(y_true)))

# Load the pretrained embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
with open ('train2/model.json', 'r') as f:
    model_params = json.load(f)
tags=model_params['tags']
def model(sentence):
    global model_params
    hidden_layer=model_params['hidden_layer']
    output_layer=model_params['output_layer']
    bh=model_params['bh']
    by=model_params['by']
    # Embed the sentence
    embedding = embedder.encode([sentence])[0]
    # Convert to list of floats for your scratch NN
    in1=embedding.tolist()

    #hidden layer
    hidden_out=[]
    for i in range(len(hidden_layer)):
        hidden_out.append(0.0)

    for i in range(len(hidden_layer)):
        for j in range(len(hidden_layer[i])):
            hidden_out[i]+=hidden_layer[i][j]*in1[j]
        hidden_out[i]=relu(hidden_out[i] + bh[i])


    #output layer
    nn_out=[]
    for i in range(len(output_layer)):
        nn_out.append(0.0)

    for i in range(len(output_layer)):
        for j in range(len(output_layer[i])):
            nn_out[i]+=output_layer[i][j]*hidden_out[j]    
        nn_out[i]=nn_out[i]+by[i]
    nn_out=softmax(nn_out)
    return nn_out
def predict_with_tags(sentence):
    final_out=[]
    final_tags=[]
    for i in range(len(sentence)):
        final_out.append(str(sentence[i]))
        final_tags.append(tags[i])
    return final_out, final_tags
def classify(sentence):
    nn_out=model(sentence)
    final_out, final_tags = predict_with_tags(nn_out)
    prob=max(final_out)
    max_index=final_out.index(prob)
    tag=final_tags[max_index]
    return tag, prob

while True:
    print(classify(input("Enter your message: ")))