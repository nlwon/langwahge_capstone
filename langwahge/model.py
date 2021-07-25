from mynn.layers.dense import dense
from mygrad.nnet.initializers import glorot_normal
import mygrad as mg
import numpy as np

class Model:
    def __init__(self, input_dim, output_dim):
        """ 
        Initializes all of the encoder and decoder layers in our model, setting them
        as attributes of the model.
        
        Parameters
        ----------
        context_words : int
            The number of context words included in our vocabulary
            
        d : int
            The dimensionality of our word embeddings

        Returns
        -------
        None
        """
        self.encoder = dense(input_dim, output_dim, weight_initializer = glorot_normal, bias = False)
    
    def __call__(self, x):
        """
        Passes data as input to our model, performing a "forward-pass".
        
        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.
        
        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(512,)
            A batch of data consisting of a resnet image dvector of shape 512
                
        Returns
        -------
        mygrad.Tensor, shape=(M, context_words)
            The result of passing the data through both the encoder and decoder.
        """
        unit_vectors = []
        for i in x:
            normal_this = self.encoder(i)
            # normal = mg.sqrt((normal_this**2).sum(keepdims = True))
            normal = normal_this / (mg.sqrt(mg.sum(normal_this ** 2, axis=1, keepdims=True)))
            unit_vectors.append(normal)
        return np.array(unit_vectors)

    @property
    def parameters(self):
        """ 
        A convenience function for getting all the parameters of our model.
        This can be accessed as an attribute, via `model.parameters` 
        
        Parameters
        ----------
        None

        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model
        """
        return self.encoder.parameters

def loss_accuracy(sim_match, sim_confuse, margin, triplet_count=0):
    """ 
    returns loss and accuracy
    
    Parameters
    ----------
    sim_match, sim_confuse, threshold, triplets = 0

    Returns
    -------
    Tuple[Tensor, ...]
        tuple of loss and accuracy 
    """
    loss = mg.nnet.losses.margin_ranking_loss(sim_match, sim_confuse, 1, margin)  
    
    flat_sim_match = sim_match.flatten()                                 
    flat_sim_confuse = sim_confuse.flatten()
    
    true_count = sum([flat_sim_match[i] > flat_sim_confuse[i] for i in range(len(flat_sim_match))])             
    
    acc = true_count / triplet_count

    return loss, acc

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