from pathlib import Path
from gensim.models import KeyedVectors
from collections import Counter
import json 
import re, string
import pickle
import numpy as np
import random
import mygrad as mg

class Coco:
    punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
    
    def get_idf(self):
        """ 
        Given the vocabulary, and the word-counts for each document, computes
        the inverse document frequency (IDF) for each term in the vocabulary.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        nt: dict{string, float}
            A dictionary storing the IDF for each term in vocab
        """
        nt = {}
        N = len(self.counters)
        for t in self.vocab:
            total = 0
            for counter in self.counters:
                if t in counter:
                    total += 1

            nt[t] = np.log10(N / total)

        return nt
    
    def strip_punc(self, corpus):
        """ 
        Removes all punctuation from a string.

        Parameters
        ----------
        corpus : str

        Returns
        -------
        str
            the corpus with all punctuation removed
        """
        # substitute all punctuation marks with ""
        
        return self.punc_regex.sub('', str(corpus))
    
    def to_counter(self, doc):
        """ 
        Produce word-count of document, removing all punctuation
        and making all the characters lower-cased.
        
        Parameters
        ----------
        doc : str
        
        Returns
        -------
        collections.Counter
            lower-cased word -> count
        """
        return Counter(self.strip_punc(doc).lower().split())

    def to_vocab(self, counters, k=None):
        """ 
        Convert a collection of counters to a sorted list of the top-k most common words 
        
        Parameters
        ----------
        counters : Sequence[collections.Counter]
            A list of counters; each one is a word tally for a document
        
        k : Optional[int]
            If specified, only the top-k words are returned
            
        Returns
        -------
        List[str]
            A sorted list of the unique strings.
        """
        vocab = Counter()
        for counter in counters:
            vocab.update(counter)
            
        return sorted(i for i,j in vocab.most_common(k))

    def __init__(self): 
        """
        load COCO metadata (json file ["images"] ["annotations"])
        load glove data (dictionary {word : word_embedding})
        load in resnet data from resnet18_features.pkl (dictionary {img id : dvector})
        
        initialize the following attributes:
        image-ID -> [cap-ID-1, cap-ID-2, ...]
        caption-ID -> image-ID
        caption-ID -> caption (e.g. 24 -> "two dogs on the grass")
        
        initialize vocab list and counters list as attributes
       
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # load COCO metadata
        with Path(r"INPUT FILEPATH").open() as f:
            self.coco_data = json.load(f)
        
        # load GloVe-200 embeddings
        self.glove = KeyedVectors.load_word2vec_format(r"INPUT FILEPATH", binary=False)

        # load image descriptor vectors
        with Path(r"INPUT FILEPATH").open('rb') as f:
            self.resnet18_features = pickle.load(f)
        
        self.imgid_to_capid = {}
        self.capid_to_imgid = {}
        self.capid_to_capstr = {}
        self.counters = []

        for caption in self.coco_data["annotations"]:
            # input caption_id to imgid_to_capid if the img_id key exists
            if self.imgid_to_capid.__contains__(caption["image_id"]):
                self.imgid_to_capid[caption["image_id"]].append(caption["id"])
            # else create new img_id object and create new caption_id list
            else:
                self.imgid_to_capid[caption["image_id"]] = [caption["id"]]

            # input img_id to capid_to_imgid
            self.capid_to_imgid[caption["id"]] = caption["image_id"]
            # input caption to capid_to_capstr
            self.capid_to_capstr[caption["id"]] = caption["caption"]
    
            self.counters.append(self.to_counter(caption["caption"]))

        self.vocab = self.to_vocab(self.counters)
        
    def random_pair(self):
        """
        returns a random caption_string and respective image id

        Parameters
        ---------
        None
        
        Returns 
        -------
        Tuple(int, string)
            this contains the random caption_string and respective image id

        """    
        # random respective caption string
        captions = self.coco_data["annotations"]
        i = random.randint(0, len(captions)) 
        
        caption_info = captions[i]
        
        return caption_info["image_id"], caption_info["caption"]
        
    def vectorize_image(self, image_id):
        """
        takes in an image_id and returns the descriptor vector of the image

        Parameters
        ---------
        image_id: int
            unique integer ID for the image in coco_data
        
        Returns 
        -------
        image_dvector: np.array shape-(512,)
            a descriptor vector of the image as provided by RESNET
        """
        if image_id not in self.resnet18_features.keys():
            return np.zeros((512,))
        else:
            return self.resnet18_features[image_id]

    def embed_text(self, text_string):
        """
        returns normal_text_embedding

        Parameters
        ---------
        text_string: String
            a caption/query text 
        
        Returns 
        -------
        String
            normal text embedding
        """
        # returns normal_text_embedding
        # embed any caption / query text using GloVe-200 embeddings weighted by IDFs of words across captions (pass in either a user's query or existing caption)     
        # lowercase, remove punc, tokenize
        text_string = strip_punc(text_string).lower().split()

        text_embedding = []
        idf = self.get_idf()
        for word in text_string:
        # check if each word in given string is in glove[], if not then embedding vector 0
        # else get glove vector (200,) for it
            glove_vector = 0
            if word in self.glove:
                glove_vector = self.glove[word]
            
            idf_word = idf[word]
            text_embedding.append(glove_vector * idf_word)

        # add all together for the final phrase embed vector, then normalize
        normal_text_embedding = mg.sqrt(mg.einsum("ij, ij -> i", text_embedding, text_embedding)).reshape(-1, 1)

        # return normal_text_embedding
        return normal_text_embedding

    def get_data(self):
        """
        returns coco_data, glove , resnet18_features, imgid_to_capid, capid_to_imgid, capid_to_capstr, counters

        Parameters
        ---------
        None

        Returns 
        -------
        Tuple(dict, dict, dict)
            contains coco data, glove data, and resnet data
        """
        return (self.coco_data, self.glove, self.resnet18_features, self.imgid_to_capid, self.capid_to_imgid, self.capid_to_capstr, self.counters)