# langwahge

# Simple Python Project Template

The basics of creating an installable Python package.

To install this package, in the same directory as `setup.py` run the command:

```shell
pip install -e .
```

This will install `example_project` in your Python environment. You can now use it as:

```python
from example_project import returns_one
from example_project.functions_a import hello_world
from example_project.functions_b import multiply_and_sum
```

To change then name of the project, do the following:
   - change the name of the directory `example_project/` to your project's name (it must be a valid python variable name, e.g. no spaces allowed)
   - change the `PROJECT_NAME` in `setup.py` to the same name
   - install this new package (`pip install -e .`)

If you changed the name to, say, `my_proj`, then the usage will be:

```python
from my_proj import returns_one
from my_proj.functions_a import hello_world
from my_proj.functions_b import multiply_and_sum
```

You can read more about the basics of creating a Python package [here](https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/Modules_and_Packages.html).


# Planning

[WORKING] (X) COCO.PY: Create a class that organizes all of the COCO data. It might store the following

    [WORKING] (X) init(), initialize coco_data, glove, and resnet18_features
        *load in coco data from captions_train2014.json (["images"] id/url/shape & /["annotations"] id/image_id/string)
        *load in resnet data from resnet18_features.pkl (image dvectors)
        *load in glove data from glove.6B.200d.txt.w2v (word embeddings)
        initialize dictionaries imgid_to_capid, capid_to_imgid, capid_to_capstr
        initialize lists counters and vocab
        
        [WORKING] (X) to_vocab()
            returns list of vocab given a list of counters
        
        [WORKING] (X) to_counter()
            returns counter of a given string
    
    [WORKING] (X) random_pair(), returns a random image_id and respective caption_string
        *get random image id and respective caption string

    [WORKING] (X) vectorize_image(image_id), returns image_dvector
        *image id --> image dvector
        if image is not in resnet features, return array of 0s

    [WORKING] (X) embed_text(text_string), returns normal_text_embedding
        *embed any caption / query text using GloVe-200 embeddings weighted by IDFs of words across captions (pass in either a user's query or existing caption)     
        *lowercase, remove punc, tokenize
        *check if each word in given string is in glove[], if not then embedding vector 0
        *else get glove vector (200,) for it
        *find idf
        *dot prod glove vector and IDF for each word
        *add all together for the final phrase embed vector, then normalize
        *return normal_text_embedding

        [WORKING] (X) strip_punc()
            used in embed_text to strip a string's punctuation

    [WORKING] (X) get_idf(), returns idf
        *part of embed_text
        creates a dictionary {vocab_word : idf_value} using vocab list and counters list

    [WORKING] (X) get_data(), returns tuple of data files
        *return coco_data, glove, resnet18_features, imgid_to_capid, capid_to_imgid, capid_to_capstr, counters

[WORKING] (X) MODEL.PY: The model class 

    [WORKING] (X) Create a MyNN model for embedding image descriptors: d⃗ img→w^img
    
        [WORKING] (X) init() 
            dense layer no bias, forward pass requires no activation func

        [WORKING] (X) call()
            Extract sets of (caption-ID, image-ID, confusor-image-ID) triples (training and validation sets) [Bhargav]
        
        [WORKING] (X) parameters() 
            return trainable params

    [WORKING] (X) loss_accuracy(sim_match, sim_confuse, threshold, triplets), returns loss and accuracy
        loss = max(0, threshold - (sim_match - sim_confuse))
        accuracy = #sim_match>sim_confuse / number of triplets
       
    [WORKING] (X) save_weights(model)
        *function to save the model weights into a file

    [WORKING] (X) load_weights(weights)
        *given a filename, load in model weights from that file

TRAINING.MD: Training the model in a jupyter notebook

    (X) Make triplets 

    (X) create_sets()
        *separate out image IDs into distinct sets for training and validation

    (X) get the caption embedding (from database)
        
        *use embed_text(caption_embedding) for labels of images
    
    (X) embed the “true” image (through model)

    (X) embed the “confusor” image (through model)
                    
    (X) compute similarities (caption and good image, caption and bad image)
        *sim_match = w_caption*w_img
        *sim_confuse = w_caption*w_img_confuse
        *delta threshold

    (X) compute margin-ranking loss and accuracy
        *loss_accuracy()

    (X) take optimization step
        *uses sgd optimizer

(X) DATABASE.PY: Create image database by mapping image feature vectors to semantic embeddings with the trained model

    (X) initialize_database()
        *initialize database as a python dictionary

    (X) load_dictionary(filepath)
        *load from pkl file given filepath

    (X) save_dictionary(filepath)
        *save a pkl file to given filepath

    (X) populate_image_database() 
        *embeddings from passing into model, normalize
        *populate database with {image_dvector: embeddings}

    (X) query_database()
        *query database with user's input
        *dot product, find the closest match

MAIN PROGRAM: Write function to query database with a caption-embedding and return the top-k images

    >>> INCOMPLETE
    query function

    >>> INCOMPLETE
    function to display set of images given COCO image ids