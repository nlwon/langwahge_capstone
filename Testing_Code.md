# training code to be implemented in a jupyter notebook alongside the main program
# saved temporarily as a text file for readability



# IMPORTS
from langwahge import * 
from mynn.optimizers.sgd import SGD
import numpy as np
import mygrad as mg
%matplotlib notebook



# LOAD IN DATA
data = Coco()
coco_data, glove, resnet18_features, imgid_to_capid, capid_to_imgid, capid_to_capstr, vocab_counts, idfs_dict = data.get_data()



# NOGGIN SETUP
from noggin import create_plot
plotter, fig, ax = create_plot(metrics=["loss", "accuracy"],max_fraction_spent_plotting=.75)



# TRIPLETS
triplets = []
for _ in range(80000):
    # generate two random pairs to compose this triplet
    ran_img_id, ran_cap_id = data.random_pair()
    confusor_img_id, confusor_cap_id = data.random_pair()
    
    # in case random and confusor images end up the same
    while ran_img_id == confusor_img_id:
        confusor_img_id, confusor_cap_id = data.random_pair()
        
    # append new triplet
    triplets.append((ran_cap_id, ran_img_id, confusor_img_id))



# SPLITTING

# 4/5 and 1/5
split_at = 0.8 
split = int(len(triplets) * split_at)

# 4/5 training
train_triplets = np.array(triplets[:split]) 

# 1/5 validation
valid_triplets = np.array(triplets[split:])



# TRAINING

# initialize model and optimizer
model = Model(512, 200)
optim = SGD(model.parameters, learning_rate = 1e-3, momentum =0.9)

# among training triplets, compile list of caption IDs
captionidlist = [i[0] for i in train_triplets]
# also get their respective string counterparts
list_phrases = [capid_to_capstr[i] for i in captionidlist]
# and use those strings to get caption embeddings
caption_embeddings = np.array([data.embed_text(i) for i in list_phrases])

# among validation triplets, compile list of caption IDs
v_captionidlist = [i[0] for i in valid_triplets]
# also get their respective string counterparts
v_list_phrases = [capid_to_capstr[i] for i in v_captionidlist]
# an use those strings to get caption embeddings
v_caption_embeddings = np.array([data.embed_text(i) for i in v_list_phrases])

# batches and epochs
batch_size = 32
num_epochs = 30

# main training loop
for epoch in range(num_epochs):
    
# batch control
    indexes = np.arange((len(train_triplets)))
    np.random.shuffle(indexes)
    for batch_count in range(0, len(train_triplets) // batch_size):
        batch_indexes = indexes[batch_count*batch_size : (batch_count + 1) * batch_size]
        img_ids = [i[1] for i in train_triplets[batch_indexes]]
        
# get normal "predictions" for this batch
        img_batch = []
        for img_id in img_ids: 
# in case of rare coco img id w/ no resnet dvector
            if img_id in data.resnet18_features.keys(): 
                img_batch.append(data.vectorize_image(img_id))
        img_preds = model(mg.tensor(img_batch))
        
# get confuser "predictions" for this batch
        conf_ids = [j[2] for j in train_triplets[batch_indexes]]
        conf_batch = []
        for conf_id in conf_ids:
# in case of rare coco img id w/ no resnet dvector
            if conf_id in data.resnet18_features.keys(): 
                conf_batch.append(data.vectorize_image(conf_id))
        conf_preds = model(mg.tensor(conf_batch))
        
# get caption_embeddings for this batch
        caption_embeddings = caption_embeddings[batch_indexes]
        
        sim_match = (caption_embeddings*img_preds).sum(axis = -1)
        sim_confuse = (caption_embeddings*conf_preds).sum(axis = -1)
        
# compute loss and accuracy (double-check model.py)
        loss, acc = loss_accuracy(sim_match, sim_confuse, 0.25, triplet_count=len(train_triplets))
        loss.backward()
        
        optim.step()
        plotter.set_train_batch({"loss" : loss.item(), "accuracy" : acc}, batch_size=batch_size)

# save weights from the trained model
filename = save_weights(model)



















































from langwahge.model import Model, loss_accuracy
from mynn.optimizers.sgd import SGD
import numpy as np
from .coco import *
import random

def save_weights(model):
    """ 
    Saves the weights from the trained model

    Parameters
    ----------
    model : obj
        the trained model that is an instance of the Model class

    Returns
    -------
    str
        the file name in which the weights are stored
    """
    np.save("weights.npy", model.parameters)
    return "weights.npy"

def load_weights(weights):
    """ 
    Loads the weights from the trained model

    Parameters
    ----------
    weights : str
        the file name in which the weights are stored

    Returns
    -------
    np.array
        loading in the saved weight matrix from the given file name
    """
    weight = np.load(weights)
    return weight

from noggin import create_plot
plotter, fig, ax = create_plot(metrics=["loss", "accuracy"],max_fraction_spent_plotting=.75)
data = Coco()
_, glove, resnet18_features, imgid_to_capid, capid_to_imgid, capid_to_capstr, _ = data.get_data()

triplets = []

# (caption_id, img_id, confuser_id)
for key in list(capid_to_imgid.keys()):
    caption_id = key
    img_id = capid_to_imgid[key]
    #values = np.array(list(imgid_to_capid.values()))
    conf_id = random.choice(list(capid_to_imgid.values()))
    if conf_id == img_id:
        conf_id = random.choice(list(capid_to_imgid.values()))
    triplets.append((caption_id, img_id, conf_id))
    #print(triplets)
    
#split the data
split_at = 0.8
split = int(len(triplets) * split_at)
train_triplets = triplets[:split] 
test_triplets = triplets[split:]

model = Model(512, 200)
optim = SGD(model.parameters, learning_rate = 1e-3, momentum =0.9)

captionidlist = [i[0] for i in train_triplets]
list_phrases = [capid_to_capstr[i] for i in captionidlist]

w_captions = np.array([data.embed_text(i) for i in list_phrases])

captionidlisttest = [i[0] for i in test_triplets]
list_phrasestest = [capid_to_capstr[i] for i in captionidlisttest]

w_captionstest = np.array([data.embed_text(i) for i in list_phrasestest])

batch_size = 32
num_epochs = 30
for epoch in range(num_epochs):
    indexes = np.arange((len(train_triplets)))
    np.random.shuffle(indexes)
    for batch_count in range(0,len(train_triplets)//batch_size):
        batch_indexes = indexes[batch_count*batch_size: batch_count*(batch_size+1)]
        
        img_ids = [i[1] for i in train_triplets[batch_indexes]]
        
        img_batch = [data.vectorize_image(imgid) for imgid in img_ids]
        img_preds = model(np.array(img_batch))
        
        conf_ids = [j[2] for j in train_triplets[batch_indexes]]
        
        conf_batch = [data.vectorize_image(confid) for confid in conf_ids]
        conf_preds = model(np.array(conf_batch))
        #print(batch)
        #w_captions = data.embed_text(capid_to_capstr[train_triplets[batch_indexes][0]])  #should correspond to the vectors 
        #confuser = model(resnet18_features[random.choice(list(resnet18_features.keys())[:82600])])  
        #w_captions = data.embed_text(np.array([capid_to_capstr[i] for i in train_triplets[batch_indexes][0]]))
        
        # captionidlist = [i[0] for i in train_triplets[batch_indexes]]
        # list_phrases = [capid_to_capstr[i] for i in captionidlist]

        # w_captions = np.array([data.embed_text(i) for i in list_phrases])
        w_captions = w_captions[batch_indexes]

        sim_match = (w_captions*img_preds).sum(axis = -1)
        sim_confuse = (w_captions*conf_preds).sum(axis = -1)
        loss, acc = loss_accuracy(sim_match, sim_confuse, 0.25, train_triplets[batch_indexes])
        
        loss.backward()
        
        optim.step()
        
        plotter.set_train_batch({"loss" : loss.item(), "accuracy" : acc}, batch_size=batch_size)

filename = save_weights(model)