from noggin import create_plot
plotter, fig, ax = create_plot(metrics=["loss"], max_fraction_spent_plotting=.75)

batch_size = 32

data = coco_data()
coco_data, glove, resnet18_features, imgid_to_capid, capid_to_imgid, capid_to_capstr, counters = data.get_self()
#split the data
split_at = 0.75
keyslist = list(resnet18_features.keys())
split = int(len(keyslist)*split_at)
training_keys = keyslist[:split]      #if this doesn't work for some reason could just hardcode

training_vectors = [resnet18_features[i] for i in training_keys]

training_captions_id = [imgid_to_capid[i] for i in training_keys]

training_captions = [capid_to_capstr[i] for i in training_captions_id]

# test_vectors = resnet18_features[split:82600]
# test_captions = 

testing_keys = keyslist[split:82600]

testing_vectors = [resnet18_features[i] for i in testing_keys]

testing_captions_id = [imgid_to_capid[i] for i in testing_keys]

testing_captions = [capid_to_capstr[i] for i in testing_captions_id]

match = 0
triplets = 0
for epoch in range(10000):
    indexes = np.arange((len(training_vectors)))
    np.random.shuffle(indexes)
    for batch_count in range(0,len(training_vectors)//batch_size):
        batch_indexes = indexes[batch_count*batch_size: batch_count*(batch_size+1)]
        batch = training_vectors[batch_indexes]  
#         print(batch)
        w_caption = training_captions[i]  #should correspond to the vectors
        prediction = model(batch)
        
        confuser = model(resnet18_features[random.choice(list(resnet18_features.keys())[:82600])])  
        
        sim_match = w_caption@prediction
        sim_confuse = w_caption@confuser
        
        if sim_match>sim_confuse:  #accuracy?
            match+=1
        triplets+=1
        
        loss = loss_accuracy(sim_match, sim_confuse, 0.25)
        
        loss.backward()
        
        optim.step()
        
        #acc =  np.mean(np.argmax(prediction, axis=1) == batch)
        
        plotter.set_train_batch({"loss" : loss.item()
                                 },
                                 batch_size=batch_size)
accuracy = match/triplets