import tensorflow as tf
import numpy as np

PAD_TOKEN = '<null>'
UNK_TOKEN = '<unk>'


NP_INT_TYPE = np.int32
NP_FLOAT_TYPE = np.float32

TF_INT_TYPE = tf.int32
TF_FLOAT_TYPE = tf.float32

""" 
    In the dictionaries below, all keys beginning with 'n_' are
    size inicator variable. Below:
        n_sentence is the length of the sentence.
        n_description is the length of the ontology description.
        n_truth is the dimensionality of the truth vector for cross entropy training.
    *** All size indicators MUST begin with 'n_'

    NP_... should be a dictionary of numpy types for each feature in each example.
    TF_... should be a dictionary of tensorflow types for each feature in each example.
    ^^ These two coincide with one another (any entry in one should have a corresponding entry in the other).

    READER_FEAT_KEY_TYPE_MAP is a dictionary of type to read in the serialized examples.
        Any numpy array that is saved should be assigned a tf.string type.
        Any integer or float that is saved should be assigned a tf.int## or tf.float## type respectively.
"""

NP_FEAT_KEY_TYPE_MAP = {
                        'sentence': NP_INT_TYPE,
                        'n_sentence': NP_INT_TYPE,
                        'description': NP_INT_TYPE,
                        'n_description': NP_INT_TYPE,
                        'truth': NP_INT_TYPE,
                        'n_truth': NP_INT_TYPE
                       }

TF_FEAT_KEY_TYPE_MAP = {
                        'sentence': TF_INT_TYPE,
                        'n_sentence': TF_INT_TYPE,
                        'description': TF_INT_TYPE,
                        'n_description': TF_INT_TYPE,
                        'truth': TF_INT_TYPE,
                        'n_truth': TF_INT_TYPE
                       }

READER_FEAT_KEY_TYPE_MAP = {
                            'sentence': tf.string,
                            'n_sentence': tf.int64,
                            'description': tf.string,
                            'n_description': tf.int64,
                            'truth': tf.string,
                            'n_truth': tf.int64
                           }

DATA_SHAPES = {
               'sentence': [None],
               'description': [None],
               'truth': [1827]
               }
