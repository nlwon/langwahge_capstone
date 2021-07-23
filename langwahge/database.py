#pickling
import pickle
import mygrad as mg
import numpy as np
# from facenet_models import FacenetModel # assume facenet_models is already installed in conda environment
# from camera import take_picture
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# import skimage.io as io
import pathlib 
import numpy as np
from .coco_data import *

def initialize_database(): 
    """
    Initalizes a dictionary database 

    Parameters
    ----------
    None
    
    Returns
    ------
    database : dict
        initialized database
    """
    database = {}
    dictionary_type = int(input("Enter 0 to input a pickled dictionary, Enter 1 to have it initialized: "))
    # Pickled Dictionary
    if dictionary_type == 0:
        file_path = input("Enter the file path and file name to the dictionary: ")
        database = load_dictionary(file_path)
    # We initialized 
    elif dictionary_type == 1:
        pass
    # Invalid Option
    else:
        print("Error: Invalid Option") 
    return database


"""def file_image(path_to_image):
    # shape-(Height, Width, Color)
    image = io.imread(str(path_to_image))
    print(image)
    if image.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        image = image[..., :-1]  # png -> RGB

def get_image(): 
    \"""
    returns image data
    ----------
    Parameters:
    None
    
    Returns
    -------
    3D numpy array

    Notes
    -----
    
\"""
    image_type = int(input("Enter 0 to load an image file, Enter 1 to take a picture with your webcam: "))
    # Image File
    if image_type == 0:
        filename = input("What's the name of the desired image file? (Include file path and extension): ")  #there's an issue with loading files in
        
        print(filename)
        image = file_image(filename) 
    # Webcamera Sample
    elif image_type == 1:
        image = take_picture()
    # Invalid Option
    else:
        print("Error: Invalid Option")  
    return image"""

    
def load_dictionary(file_path): 
    """
    loads a dictionary from a Pickle file
    Parameters
    ----------
    file_path: string
        path and name of file
    
    Returns
    -------
    dictionary 
        unpickled dictionary

    Notes
    -----
    
    """
    with open(file_path, mode = "rb") as opened_file:
        return pickle.load(opened_file)


def save_dictionary(dict, file_path): 
    """
    saves a dictionary to a Pickle file
    Parameters
    ----------
    dict: dictionary
        dictionary to pickle
    file_path: string
        path and name of file to store dictionary to 
    Returns
    -------
    None
    
    Notes
    -----
    
    """
    with open(file_path, mode = "wb") as opened_file:
        pickle.dump(dict, opened_file)


def populate_image_database(image, database): 
    """
    embeddings from passing into model, normalize
        populate database with {image_dvector: embeddings}

    Parameters
    --------
    image_dvector: np.ndarray
        descriptor vector of image
    text_string:
        text corresponding to the image (?) not sure but we need it
    database:
        image database
    Returns 
    -------
    None

    """
    coco_data = coco_data()
    caption_string = coco_data.capid_to_capstr
    image_dvector = vectorize_image(image_id)
    normal_text_embedding = embed_text(caption_string)
    database[image_dvector] = normal_text_embedding



def query_database(text): 
     """

     Parameters
     ---------

     Returns 
     -----
     """
     


