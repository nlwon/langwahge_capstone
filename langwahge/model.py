from mynn.layers.dense import dense
from mygrad.nnet.initializers import glorot_normal
import random

class Model:
    def __init__(self, input_dim, output_dim):
        """ Initializes all of the encoder and decoder layers in our model, setting them
        as attributes of the model.
        
        Parameters
        ----------
        context_words : int
            The number of context words included in our vocabulary
            
        d : int
            The dimensionality of our word embeddings
        """
        # STUDENT CODE HERE
        self.encoder = dense(input_dim,output_dim, weight_initializer = glorot_normal, bias = False)
    
    def __call__(self, x):
        ''' Passes data as input to our model, performing a "forward-pass".
        
        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.
        
        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(M, context_words)
            A batch of data consisting of M words from the context matrix,
                each tracking the number of co-occurences with `context_words` words.
                
        Returns
        -------
        mygrad.Tensor, shape=(M, context_words)
            The result of passing the data through borth the encoder and decoder.
        '''
        # STUDENT CODE HERE
        normal_this = self.encoder(x)
        normal = mg.sqrt((normal_this**2).sum(keepdims = True))
        return normal 

    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.
        
        This can be accessed as an attribute, via `model.parameters` 
        
        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model"""
        # STUDENT CODE HERE
        return self.encoder.parameters

def loss_accuracy(sim_match, sim_confuse, threshold, triplets = 0):
    loss = max(0, threshold - (sim_match - sim_confuse))
    #list_to_sum = [1 if sim_match>sim_confuse else 0 for i in triplets]
    #acc = sum(list_to_sum)/len(triplets)
    return loss#, acc

model = Model(input_dim = 512, output_dim = 200)
optim = SGD(model.parameters, learning_rate = 1e-3, momentum = 0.9)
from noggin import create_plot
plotter, fig, ax = create_plot(metrics=["loss"], max_fraction_spent_plotting=.75)

batch_size = 32

data = coco_data()
coco_data, glove, resnet18_features, imgid_to_capid, capid_to_imgid, capid_to_capstr, counters = data.get_self()
#split the data
training_vectors = resnet18_features[:61959]
test_vectors = resnet18_features[61959:82600]

for epoch in range(10000):
    indexes = np.arange((len(training_vectors)))
    np.random.shuffle(indexes)
    for batch_count in range(0,len(training_vectors)//batch_size):
        batch_indexes = indexes[batch_count*batch_size: batch_count*(batch_size+1)]
        batch = training_vectors[batch_indexes]  
#         print(batch)
        prediction = model(batch)
        
        confuser = model(resnet18_features[random.choice(list(resnet18_features)[:82600])])  
        
        sim_match = w_caption@prediction
        sim_confuse = w_caption@confuser
        
        loss = loss_accuracy(sim_match, sim_confuse, 0.25)
        
        loss.backward()
        
        optim.step()
        
        #acc =  np.mean(np.argmax(prediction, axis=1) == batch)
        
        plotter.set_train_batch({"loss" : loss.item()
                                 },
                                 batch_size=batch_size)