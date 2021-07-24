from .coco import *
import pickle

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
    # Input existing dictionary
    if dictionary_type == 0:
        file_path = input("Enter the file path and file name to the dictionary: ")
        database = load_dictionary(file_path)
    # Initialize a dictionary
    elif dictionary_type == 1:
        pass
    # Invalid Option
    else:
        print("Error: Invalid Option") 
    return database
    
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
    """
    with open(file_path, mode = "wb") as opened_file:
        pickle.dump(dict, opened_file)

def populate_image_database(img_dvector_batch, img_embedding_batch, database): 
    """
    maps image dvectors to image embeddings

    Parameters
    --------
    img_dvector_batch, img_embedding_batch, database
        dvectors, respective embeddings, and the database

    Returns 
    -------
    populated database
    """
    database = dict(zip(img_dvector_batch, img_embedding_batch))
    return database

def query_database(query, database, k): 
    """
    query the database with a user input

    Parameters
    ---------
    query : str
        user's input
    
    database : dict
        database to search within

    k : int
        return the k most relevant results

    Returns 
    -----
    list of k most relevant image ids, ordered by relevancy
    """
    embedded_query = Coco.embed_text(query)
    topk = sorted(embedded_query @ database.values())
    return topk[0:k]

