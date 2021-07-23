from langwahge.model import Model, loss_accuracy
from mynn.optimizers.sgd import SGD
import numpy as np
import coco_data
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

# from noggin import create_plot
# plotter, fig, ax = create_plot(metrics=["loss"], max_fraction_spent_plotting=.75)

data = coco_data()
_, glove, resnet18_features, imgid_to_capid, capid_to_imgid, capid_to_capstr, _ = data.get_data()

triplets = []
# (caption_id, img_id, confuser_id)
for key, value in capid_to_imgid:
    caption_id = key
    img_id = value
    conf_id = random.choice(list(imgid_to_capid.keys()))
    triplets.append((caption_id, img_id, conf_id))

#split the data
split_at = 0.8
split = int(len(triplets) * split_at)
train_triplets = triplets[:split] 
test_triplets = triplets[split:]

model = Model(512, 200)
optim = SGD(model.parameters, learning_rate = 1e-3, momentum =0.9)

batch_size = 32
num_epochs = 10000
for epoch in range(num_epochs):
    indexes = np.arange((len(train_triplets)))
    np.random.shuffle(indexes)
    for batch_count in range(0,len(train_triplets)//batch_size):
        batch_indexes = indexes[batch_count*batch_size: batch_count*(batch_size+1)]
        img_batch = data.vectorize_image(train_triplets[batch_indexes][1])
        img_preds = model(img_batch)
        conf_batch = data.vectorize_image(train_triplets[batch_indexes][2])
        conf_preds = model(conf_batch)
        #print(batch)
        w_captions = data.embed_text(capid_to_capstr[train_triplets[batch_indexes][0]])  #should correspond to the vectors 
        #confuser = model(resnet18_features[random.choice(list(resnet18_features.keys())[:82600])])  
        w_captions = data.embed_text(np.array([capid_to_capstr[i] for i in train_triplets[batch_indexes][0]]))
        
        list_phrases = [capid_to_capstr[i] for i in train_triplets[batch_indexes][0]]

        w_captions = np.array([data.embed_text(i) for i in list_phrases])

        sim_match = w_captions@img_preds
        sim_confuse = w_captions@conf_preds
        loss, acc = loss_accuracy(sim_match, sim_confuse, 0.25, len(train_triplets))
        
        loss.backward()
        
        optim.step()
        
        # plotter.set_train_batch({"loss" : loss.item()
        #                          },
        #                          batch_size=batch_size)

filename = save_weights(model)