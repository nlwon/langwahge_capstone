from pathlib import Path
from gensim.models import KeyedVectors
from collections import Counter
import json 
import re, string
import pickle
import numpy as np
import random

class Coco:
    punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
    
    def strip_punc(self, corpus):
        """ 
        removes all punctuation from a string

        Parameters
        ----------
        corpus : str

        Returns
        -------
        the corpus with all punctuation removed (str)
        """
        return self.punc_regex.sub('', str(corpus))
    
    def to_counter(self, doc):
        """ 
        produces word counter of document, while removing punctuation and translating to lowercase
        
        Parameters
        ----------
        doc : str
        
        Returns
        -------
        lowercase-word : count (counter)
        """
        return Counter(self.strip_punc(doc).lower().split())

    def to_vocab(self, counters, k=None):
        """ 
        convert a collection of counters to a sorted list of the top-k most common words 
        
        Parameters
        ----------
        counters : sequence[collections.Counter]
            list of counters; each one is a word tally for a document
        
        k : optional[int]
            if specified, only the top-k words are returned
            
        Returns
        -------
        a sorted list of the unique strings (list(str))
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
        
        calculate idfs, and map vocabwords : idfs for use in embedding function
       
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # load COCO metadata
        with Path(r"langwahge\data\captions_train2014.json").open() as f:  
            self.coco_data = json.load(f)
        
        # load GloVe-200 embeddings
        self.glove = KeyedVectors.load_word2vec_format(r"langwahge\data\glove.6B.200d.txt.w2v", binary=False)

        # load image descriptor vectors
        with Path(r"langwahge\data\resnet18_features.pkl").open('rb') as f:
            self.resnet18_features = pickle.load(f)
        
        self.imgid_to_capid = {}
        self.capid_to_imgid = {}
        self.capid_to_capstr = {}
        self.vocab_counts = {} # all words and counts among every caption
        
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
    
            # compile all the words and their counts among every caption
            cap_counter = self.to_counter(caption)
            for word in cap_counter.keys():
                if word not in self.vocab_counts:
                    self.vocab_counts[word] = 1
                elif word in self.vocab_counts:  
                    self.vocab_counts[word] += 1
        
        # log10 of (number of all captions) / (count of each word among all captions)
        N = len(self.coco_data["annotations"])
        idfs = np.log10(np.array([N / vocab_count for vocab_count in self.vocab_counts.values()]))
        # maps WORD-STRING to WORD-IDF
        self.idfs_dict = dict(zip(self.vocab_counts.keys(), list(idfs)))
        
    def random_pair(self):
        """
        returns a random caption id and respective image id

        Parameters
        ---------
        None
        
        Returns 
        -------
        random caption id and respective image id (tuple(int, int))
        """    
        # random respective caption string
        captions = self.coco_data["annotations"]
        i = random.randint(0, len(captions)-1) 
        
        caption_info = captions[i]
        
        return caption_info["image_id"], caption_info["id"]
        
    def vectorize_image(self, image_id):
        """
        takes in an image_id and returns the descriptor vector of the image

        Parameters
        ---------
        image_id: int
            unique integer ID for the image in coco_data
        
        Returns 
        -------
        a descriptor vector of the image as provided by RESNET (np.array shape-(512,))
        """
        if image_id not in self.resnet18_features: #hashtable generally doesnt need the .keys() when checking if key exists 
            return np.zeros((512,))
        else:
            return self.resnet18_features[image_id]
    
    def embed_text(self, text_string):
        """
        returns normal_text_embedding

        Parameters
        ---------
        text_string: string
            a caption/query text 
        
        Returns 
        -------
        normal text embedding 
        """
        # filter text_string with lowercase, remove punc, tokenize
        text_string = self.strip_punc(text_string).lower().split()

        # text_embedding = np.zeros((200,))
        text_embedding = [0] * 200
        counter = 0
        for word in text_string:
                              
            # if word in string is not in glove[], embedding vector is 0
            # glove_vector = np.zeros((200,))
            glove_vector = [0] * 200

            # if word in string is not in idf vocabulary, embedding vector is 0
            # word_idf = np.zeros((200,))
            word_idf = [0] * 200
            
            # if word in string is in glove[], get glove embedding vector                            
            if word in self.glove:
                glove_vector = self.glove[word]
      
            # if word in string is in idf vocabulary, get respective idf
            if word in self.idfs_dict.keys():
                word_idf = self.idfs_dict[word]

            # update text_embedding from this word's embedding vector and id
            text_embedding[counter] = (glove_vector * word_idf)
            counter += 1

        # add all together for the final phrase embed vector, then normalize
        # normalized_text_embedding = text_embedding / (np.sqrt(list(np.sum(np.power(text_embedding, 2), keepdims=True))))
        text_embedding = np.array(text_embedding)
        normalized_text_embedding = text_embedding / np.sqrt(list(np.sum((text_embedding ** 2), keepdims=True))).flatten()

        # return normal_text_embedding
        return normalized_text_embedding
    
    def get_data(self):
        """
        returns all attributes

        Parameters
        ---------
        None

        Returns 
        -------
        Tuple(dict x 8)
            contains coco_data, glove , resnet18_features, imgid_to_capid, 
            capid_to_imgid, capid_to_capstr, vocab_counts, idfs_dict
        """
        return (self.coco_data, self.glove, self.resnet18_features, self.imgid_to_capid, self.capid_to_imgid, self.capid_to_capstr, self.vocab_counts, self.idfs_dict)
