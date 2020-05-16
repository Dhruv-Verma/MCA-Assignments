from nltk.corpus import abc
import numpy as np
import pickle
import re
from collections import Counter
import random, math
import itertools
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# creating corpus
corpus = []
for text_id in abc.fileids():
    text = abc.raw(text_id)
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub('[^a-zA-Z1-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    corpus.append([w for w in text.split() if w != ''])

n_docs = len(corpus)

# subsample frequent words
filtered_corpus = []
word_counts = dict(Counter(list(itertools.chain.from_iterable(corpus))))
total_words = np.sum(list(word_counts.values()))
freq = {word: word_counts[word]/float(total_words) for word in word_counts}
threshold = 1e-5
for doc in corpus:
    filtered_doc = []
    for word in doc:
        p_word = math.sqrt(threshold/freq[word])   # probability with which word is kept
        if random.random() < p_word:
            filtered_doc.append(word) 
    filtered_corpus.append(filtered_doc)        

# creating vocabulary
vocab = set()
for x in filtered_corpus:
    vocab = vocab.union(set(x))
vocab = list(vocab)
vocab_size = len(vocab)
    
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for i, w in enumerate(vocab)}

# creating data for skip-gram model

C = 5   # Context window
data = []
for doc in filtered_corpus:
    for i in range(len(doc)):
        R = random.randint(1, C)
        word = doc[i]
        lower_idx = max(0, i-R)
        upper_idx = min(i+R, len(doc)-1)
        for j in range(lower_idx, upper_idx+1):    
            if i!=j:
                data.append((w2i[doc[i]], w2i[doc[j]]))


# negative sampling
k = 5           # sample size 
p_noise = {}    # probablity distribution of noise
Z = np.sum(np.power(list(word_counts.values()) ,0.75))
for word in word_counts:
    p_noise[word] = np.power(word_counts[word], 0.75) / Z

def get_negative_samples(batch_size, k, p_noise):
    neg_samples = []
    sampled_words = np.array(np.random.choice(a = list(p_noise.keys()), size = (batch_size, k), 
                                              p = list(p_noise.values())))
    for neg_word_list in sampled_words:
        neg_samples.append([w2i[word] for word in neg_word_list])
    
    neg_samples = torch.LongTensor(neg_samples).to(device)
    return neg_samples                         

# neg = get_negative_samples(10, 5, p_noise)

# creating the model
class Word2Vec(nn.Module):
    
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.target_emb = nn.Embedding(vocab_size, embed_dim)
        self.context_emb = nn.Embedding(vocab_size, embed_dim)
        
        # initialize embeddings
        self.target_emb.weight.data.uniform_(-1, 1)
        self.context_emb.weight.data.uniform_(-1, 1)
        
    def forward(self, target_ids, context_ids, negative_ids):
        target_vector = self.target_emb(target_ids)
        context_vector = self.context_emb(context_ids)
        negative_vector = self.context_emb(negative_ids)
        return target_vector, context_vector, negative_vector

# creating Negative Sampling Loss    
class NegativeSamplingLoss(nn.Module):
    def __int__(self):
        super().__init__()
        
    def forward(self, target_vector, context_vector, negative_vector):
        batch_size, embed_dim = target_vector.shape
        
        target_vector = target_vector.reshape(batch_size, embed_dim, 1)
        context_vector = context_vector.reshape(batch_size, 1, embed_dim)
        
        # negative log loss
        positive_loss = torch.bmm(context_vector, target_vector).sigmoid().log().squeeze()
        negative_loss = torch.bmm(-1 * negative_vector, target_vector).sigmoid().log().squeeze().sum(1)
        loss = -1 * (positive_loss + negative_loss).mean()
        
        return loss
        
# Hyperparameters
embed_dim = 128
lr = 0.001
batch_size = 1024
n_epochs = 50

# Model intitialization
model = Word2Vec(vocab_size, embed_dim).to(device)
loss_fn = NegativeSamplingLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
batches = DataLoader(data,batch_size=batch_size, shuffle=True)

loss_values = []
tsne_values = []
for e in range(n_epochs):
    print('epoch: ' + str(e)+ '/' + str(n_epochs))
    i_batch = 1
    for target_ids, context_ids in batches:
        target_ids, context_ids = target_ids.to(device), context_ids.to(device)
        len_batch = len(target_ids)
        negative_ids = get_negative_samples(len_batch, k=5, p_noise = p_noise)
        
        target_vector, context_vector, negative_vector = model(target_ids, context_ids, negative_ids)
        loss = loss_fn(target_vector, context_vector, negative_vector)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_values.append(loss.item())
        if(i_batch % 100 == 0):
            print(i_batch, loss.item())
        i_batch+=1
    
    # Visualization using TSNE
    print('calculating TSNE')
    data_viz_len = 300
    viz_embedding = model.target_emb.weight.data.cpu()[:data_viz_len]
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(viz_embedding)
    tsne_values.append(embed_tsne)
    print('done............')
    
    # plt.figure(figsize=(16,16))
    # for w in list(vocab[:data_viz_len]):
    #     w_id = w2i[w]    
    #     plt.scatter(embed_tsne[w_id,0], embed_tsne[w_id,1])
    #     plt.annotate(w, (embed_tsne[w_id,0], embed_tsne[w_id,1]), alpha=0.7)
    # plt.tight_layout()

import plotly.express as px
from plotly import offline

a=[]
for i in range(50):
    for w_id, cord in enumerate(tsne_values[i]): 
        a.append([i+1,i2w[w_id],cord[0],cord[1]])

a= pd.DataFrame(a, columns=['epoch', 'word', 'x', 'y'])
range_x = [min(a.x), max(a.x)]
range_y = [min(a.y), max(a.y)]
        
fig = px.scatter(a, x='x', y='y', animation_frame='epoch',
           color='word', range_x=range_x, range_y=range_y)
        
offline.plot(fig)        
        
