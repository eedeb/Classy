#Data prep
# Data prep
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load intents file
with open('intents2.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Collect tags and (pattern, tag) pairs
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    if tag not in tags:
        tags.append(tag)
    for pattern in intent['patterns']:
        xy.append((pattern, tag))
# Load the pretrained embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
# Prepare raw lists
raw_inputs = []
raw_outputs = []

for (pattern_sentence, tag) in xy:
    # Encode sentence to embedding vector
    embedding = embedder.encode([pattern_sentence])[0]
    raw_inputs.append(embedding)

    # One-hot encode the tag
    one_hot = [0] * len(tags)
    one_hot[tags.index(tag)] = 1
    raw_outputs.append(one_hot)
print(tags)
# Convert to numpy arrays
raw_inputs = np.array(raw_inputs)      # shape: (num_samples, embed_dim)
raw_outputs = np.array(raw_outputs)    # shape: (num_samples, num_tags)

# Transpose so it's neuron × samples
input_vector = raw_inputs.T.tolist()   # shape: (embed_dim, num_samples)
output_vector = raw_outputs.T.tolist() # shape: (num_tags, num_samples)

print("Input shape (neurons × samples):", len(input_vector), "x", len(input_vector[0]))
print("Output shape (neurons × samples):", len(output_vector), "x", len(output_vector[0]))
def sentence_to_input(sentence):
    # Embed the sentence
    embedding = embedder.encode([sentence])[0]
    # Convert to list of floats for your scratch NN
    return embedding.tolist()


#Model
'''
input_vector=[
    [0,0,0,0,1,1,1,1],
    [0,0,1,1,0,0,1,1],
    [0,1,0,1,0,1,0,1]
    ]
output_vector=[
    [0,0,0,0,0,0,0,1],
    [0,1,1,0,1,0,0,1]
    ]
    '''
hidden_neurons=8
#x1=[0,0,1,1]
#x2=[0,1,0,1]
#output=[1,0,0,1]
#output1=[1,0,0,1]

#0 1 0 1
#0 0 1 1
#0 0 0 1 


#1 0 0 1


#h1=[-6.037570626537101, -7.435811060154638]
#h2=[3.4423171046672287, -9.227337188332427]
#y1=[7.498952627675039, -2.6830349347628517]


#h1=[0.5,0.5]
#h2=[0.5,0.5]
#y1=[1.0, -2.0]
input_neurons=len(input_vector)
output_neurons=len(output_vector)

print(str(input_neurons)+'>'+str(hidden_neurons)+'>'+str(output_neurons))

hidden_layer=[]
output_layer=[]
for i in range(hidden_neurons):
    connections=[]
    for j in range(input_neurons):
        connections.append(np.random.rand()*0.01)
    hidden_layer.append(connections)
for i in range(output_neurons):
    output_connections=[]
    for j in range(hidden_neurons):
        output_connections.append(np.random.rand()*0.01)
    output_layer.append(output_connections)
bh = [0.0 for _ in range(len(hidden_layer))]
by = [0.0 for _ in range(len(output_layer))]

learning_rate=0.01



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
def forward_pass(in1,expected):
    #hidden layer
    hidden_out=[]
    hidden_pre_activation=[]
    for i in range(len(hidden_layer)):
        hidden_out.append(0.0)
        hidden_pre_activation.append(0.0)

    for i in range(len(hidden_layer)):
        for j in range(len(hidden_layer[i])):
            hidden_out[i]+=hidden_layer[i][j]*in1[j]
        hidden_pre_activation[i]=hidden_out[i] + bh[i]
        hidden_out[i]=relu(hidden_pre_activation[i])


    #output layer
    nn_out=[]
    for i in range(len(output_layer)):
        nn_out.append(0.0)

    for i in range(len(output_layer)):
        for j in range(len(output_layer[i])):
            nn_out[i]+=output_layer[i][j]*hidden_out[j]    
        nn_out[i]=nn_out[i]+by[i]
    nn_out=softmax(nn_out)
    #backpropogation
    for i in range(len(nn_out)):
        error=nn_out[i] - expected[i]
        #update hl -> ol

        for j in range(len(output_layer[i])):
            output_layer[i][j]-= learning_rate*error*hidden_out[j]
        #Update bias
        by[i] -= learning_rate * error 
    
        #update: input -> hidden layer)
    for j in range(len(hidden_layer)):
        hidden_relu_derivative=relu_derivative(hidden_pre_activation[j])

        hidden_error_sum=0.0
        for k in range(len(output_layer)):
            hidden_error_sum+=(nn_out[k] - expected[k]) * output_layer[k][j]

        hidden_error=hidden_error_sum*hidden_relu_derivative
    
        for ii in range(len(hidden_layer[j])):
            hidden_layer[j][ii]-= learning_rate*hidden_error*in1[ii]
        #Update bias
        bh[j] -= learning_rate * hidden_error
    
    return nn_out
    
def test_model(in1):
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
    for i in range(len(sentence)):
        print(tags[i]+': '+str(sentence[i]))

def save_model(filename="model.json"):
    model_data = {
        "tags": tags,
        "hidden_layer": hidden_layer,
        "output_layer": output_layer,
        "bh": bh,
        "by": by,
        "learning_rate": learning_rate,
        "hidden_neurons": hidden_neurons,
        "input_neurons": input_neurons,
        "output_neurons": output_neurons
    }
    # Convert numpy floats to plain floats for JSON
    clean_data = json.loads(json.dumps(model_data, default=lambda x: float(x)))
    with open(filename, "w") as f:
        json.dump(clean_data, f, indent=2)

passthrough=[] 
for j in range(len(input_vector)):
    passthrough.append(input_vector[j])

def train():
    epochs=100
    global forward_pass
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(input_vector[0])):  # all
            passthrough=[] 
            for j in range(len(input_vector)):
                passthrough.append(input_vector[j][i])
            passthrough_o=[] 
            for j in range(len(output_vector)):
                passthrough_o.append(output_vector[j][i])
            pred = forward_pass(passthrough, passthrough_o)
            total_loss += cross_entropy(passthrough_o, pred)
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
train()
save_model()

sentence=sentence_to_input("Hello, how are you?")
final_nn_out=test_model(sentence)
predict_with_tags(final_nn_out)