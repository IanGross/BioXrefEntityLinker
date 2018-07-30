import os
import pickle
from collections import Counter

import tensorflow as tf

import data_constants

import re
from more_itertools import unique_everseen
import nltk
import gensim 
import sys
import json
import numpy as np


PAD_TOKEN = '<null>'
UNK_TOKEN = '<unk>'
TERMS_FILE = "../dumps/word_contexts.pkl"
ONT_CONTEXTSFILE = "../dumps/unique_truthscont.pkl"
EMBEDDING_FNAME = "../w2vPubmedv1/modelWord2Vec"
ONTLABELIDMAPPINF_FNAME = "../dumps/LabelIDMapping.json"

def load_pickle(path, verbose=False):
    with open(path, 'rb') as pick_f:
        file_data = pickle.load(pick_f)
        if verbose:
            print ('Loaded %s..' %path)
        return file_data

def load_json(path, verbose=False):
    with open(path, 'rb') as json_f:
        file_data = json.load(json_f)
        if verbose:
            print ('Loaded %s..' %path)
        return file_data

def save_pickle(data, path, verbose=False):
    with open(path, 'wb') as pick_f:
        pickle.dump(data, pick_f, pickle.HIGHEST_PROTOCOL)
        if verbose:
            print ('Saved %s..' % path)


# Annotations should be a list of tokenized sentences/descriptions/text
def build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for sentence in annotations:
        words = sentence.split(' ')
        for w in words:
            counter[w] += 1

        if len(words) > max_len:
            max_len = len(sentence.split(" "))

    vocab_words = [word for word in counter if counter[word] >= threshold]
    print('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab_words), threshold))

    word_to_idx = {data_constants.PAD_TOKEN: 0, data_constants.UNK_TOKEN: 1}
    idx = len(word_to_idx)
    for word in vocab_words:
        word_to_idx[word] = idx
        idx += 1
    print('Max length of sentence: %d' % max_len)
    return word_to_idx


# If live, then w_idx_map should be the already loaded word_to_idx dictionary
# https://gist.github.com/ottokart/673d82402ad44e69df85
# https://groups.google.com/forum/#!topic/word2vec-toolkit/GFNZkoDPd0g
def get_word_embeddings_from_binary(embedding_fname, init_range=(-0.05, 0.05), w_idx_map='./',
                                    outfname='', verbose=True, live=False, embed_dim=300):
    """
        Loads 300x1 word vecs from Google (Mikolov) word2vec.
        Words in w_idx_map that do not have embeddings in word2vec embeddings are given random embeddings
            where the random values are within init_range.
        w_idx_map: A dictionary that maps each word (string) to a unique id (int). The id will correspond to the 
                    row of the embedding matrix.
    """
    if verbose:
        print('Obtaining word embeddings...')
        sys.stdout.flush()
    if not live:
        word_idx = load_pickle(w_idx_map)
    else:
        word_idx = w_idx_map

    init_range = sorted(init_range)
    a = init_range[0]
    b = init_range[1]
    count = 0
    float_size = 4

    with open(embedding_fname, 'rb') as f:
        c = None

        # read the header
        header = ''
        while c != '\n':
            c = f.read(1)
            header += c

        total_num_vectors, vector_len = (int(x) for x in header.split())
        if verbose:
            print('Number of vectors in embedding file: %d' % total_num_vectors)
            print('Vector size: %d' % vector_len)
            sys.stdout.flush()

        embedding_matrix = (b - a) * np.random.random_sample((len(word_idx), embed_dim)) + a

        i = 0
        print_every = total_num_vectors / 100
        while i < total_num_vectors:
            word = ""
            while True:
                c = f.read(1)
                if c == " ":
                    break
                word += c
            binary_vector = f.read(float_size * vector_len)
            if word in word_idx:
                index = word_idx[word]
                embedding_matrix[index, :] = np.fromstring(binary_vector, dtype=np.float32)
                count += 1
            if verbose and i and (i % print_every) == 0:
                print('\tSearched through %d out of %d vectors.' % (i, total_num_vectors))
            i += 1
    if outfname:
        save_pickle(embedding_matrix, outfname)
    if verbose:
        print('Finished obtaining word embeddings...')
        print('\tPretrained embeddings covered %d out of %d vocabulary words' % (count, len(word_idx)))
    return embedding_matrix


def load_json(path, verbose=False):
    with open(path, 'rb') as json_f:
        file_data = json.load(json_f)
        if verbose:
            print ('Loaded %s..' %path)
        return file_data

'''
Function should read in both sent and ont consolidated sentences and build a common vocabulary list
And a list of tuple data_pairs
Files: ../dumps/word_contexts.pkl, ../dumps/unique_truthconts.pkl
{'sentence': 'Hello , my name is Spencer .', 'n_sentence': 1,
            'description': 'We is mouse', 'n_description': 1,
            'truth': np.arange(D, dtype=np.float32),
            'n_truth': D}
vocab_all is a common data structure to build the vocabulary of words
vocab_dict is used to contain the struct of {'sent','n_sent','ont','n_ont','truth','n_truth'}
'''
def build_consolidatedsentandontmapping(terms_file, ont_contextsfile, labelmapping_file):
    vocab_all = set()
    ont_seens = []
    #List of tuples, sent; ont
    vocab_dict = []
    ont_descs = []

    labelidmapping = load_json(labelmapping_file)
    num_truths = len(labelidmapping)
    terms_cont = load_pickle(terms_file)
    ont_cont = load_pickle(ont_contextsfile)
    for term_neighborhood in terms_cont:
        sent_consolidated = sent_preprocessingandcleanup(term_neighborhood['consolidated'])
        associatedont_consolidated = sent_preprocessingandcleanup(ont_cont[term_neighborhood['truth']]['consolidated'])
        truth_id = int(labelidmapping[term_neighborhood['truth']])
        truth_np =np.zeros(shape=(num_truths,1),dtype=np.int64)
        truth_np[truth_id - 1] = 1
        vocab_all.add(sent_consolidated)
        vocab_all.add(associatedont_consolidated)
        vocab_dict.append({'sentence':sent_consolidated,'n_sentence':len(sent_consolidated.split(" ")),'description':associatedont_consolidated, 'n_description':len(associatedont_consolidated.split(" ")),'truth': truth_np,'n_truth':num_truths})
        '''
        Add in the IDs as a part of the tuple portion to reduce sorting complexity
        The ont descriptions need to be arranged by their order
        '''
        if truth_id not in ont_seens:
            ont_seens.append(truth_id)
            ont_descs.append(({'description':associatedont_consolidated, 'n_description': len(associatedont_consolidated.split(" ")),'truth':truth_np,'n_truth':num_truths},truth_id))
    print("Num truths ",num_truths)
    return (list(vocab_all), vocab_dict, ont_descs)


def sent_preprocessingandcleanup(sent):
    #Split on the puncts
    cleanedsent = [word.lower() for word in nltk.word_tokenize(sent) if word != ""]
    #Remove all non alphanumeric characters
    cleanedsent = re.sub(r'[^\w\s]',' '," ".join(cleanedsent))
    #Remove numbers
    cleanedsent = re.sub(r'[\d+]','',cleanedsent)
    #Replace multiple spaces
    cleanedsent = re.sub(r'\s+',' ',cleanedsent)
    #Return unique chars
    cleanedsent= " ".join(list(unique_everseen(cleanedsent.split(" "))))
    return cleanedsent

#Normalise the word embedding by tf-idf
def infoContentOfWord (w1,model): 
    # print w1
    # return -1*np.log ( model.vocab[w1].count*1.0 / model.corpus_count )
    a = np.log(model.wv.vocab[w1].count*1.0)
    b = np.log(460155)
    return 1.0 - a/b 

# Retrieve word embeddings from w2vPubmed
def get_word_embeddings_from_w2vPMd(embedding_fname, init_range=(-0.05, 0.05), w_idx_map={},
                                    outfname='../dumps/word_embeddings.pkl', verbose=True, live=True, embed_dim=250):                                     
    """
        Loads 300x1 word vecs from Google (Mikolov) word2vec.
        Words in w_idx_map that do not have embeddings in word2vec embeddings are given random embeddings
            where the random values are within init_range.
        w_idx_map: A dictionary that maps each word (string) to a unique id (int). The id will correspond to the 
                    row of the embedding matrix.
    """
    if verbose:
        print('Obtaining word embeddings...')
        sys.stdout.flush()

    #Just need to load embeddings using gensim
    print("loading gensim library, and the already trained word model from 15GB of Pubmed open access articles (may take 1-2 minutes)")
    model = gensim.models.Word2Vec.load(embedding_fname)

    if not live:
        word_idx = load_pickle(w_idx_map)
    else:
        word_idx = w_idx_map

    init_range = sorted(init_range)
    a = init_range[0]
    b = init_range[1]
    count = 0
    float_size = 4
    #Randomly intialising embedding matrix
    embedding_matrix = (b - a) * np.random.random_sample((len(word_idx), embed_dim)) + a

    i = 0
    for word in word_idx.keys():
        index = word_idx[word]
        #If it is in the vocab
        if word in model.wv.vocab:
            embedding_matrix[index, :] = infoContentOfWord(word, model)
            count+=1  
               
    if outfname:
        save_pickle(embedding_matrix, outfname)
    if verbose:
        print('Finished obtaining word embeddings...')
        print('\tPretrained embeddings covered %d out of %d vocabulary words' % (count, len(word_idx)))
    return embedding_matrix


def _int64_feature(value):
    """
        Wrapper for inserting an int64 Feature into a SequenceExample proto.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """
        Wrapper for inserting a bytes Feature into a SequenceExample proto.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tf_records(split_examples, split_name, out_dir, n_files,
                     feat_key_type, rando=True):
    adj_n_files = n_files
    save_size = np.ceil(len(split_examples) / float(n_files))
    if not (len(split_examples) / n_files):
            adj_n_files = len(split_examples)
    if rando:
        print('Writing features to files randomly')
    else:
        #HACK for now
        n_vids = len(split_examples)
        save_size = np.ceil(n_vids / float(n_files))
        if not n_vids / n_files:
            adj_n_files = n_vids
        print('Writing features to files in order')
    out_template_fname = '.'.join([split_name, 'data%d', 'tfrecord'])
    save_path = os.path.join(out_dir, out_template_fname)
    all_writers = [tf.python_io.TFRecordWriter(save_path % i)
                   for i in range(adj_n_files)]
    file_sizes = [0 for _ in range(len(all_writers))]
    ind = 0
    for ex in split_examples:
        feat_dict = {}
        for k, v in feat_key_type.items():
            if type(ex[k]) is np.ndarray:
                feat_dict[k] = _bytes_feature(ex[k].astype(v).tostring())
            else:
                feat_dict[k] = _int64_feature(ex[k])

        if rando:
            ind = np.random.choice(len(all_writers))
        elif file_sizes[ind] >= save_size:
            ind += 1

        example = tf.train.Example(features=tf.train.Features(feature=feat_dict))
        all_writers[ind].write(example.SerializeToString())
        file_sizes[ind] += 1

    for writer in all_writers:
        writer.close()


if __name__ == '__main__':
    # data = [
    #         {'sentence': 'Hello , my name is Spencer .', 'n_sentence': 1,
    #         'description': 'We is mouse', 'n_description': 1,
    #         'truth': np.arange(D, dtype=np.float32),
    #         'n_truth': D},

    #         {'sentence': 'We are doing Biomedical Entity Linking .', 'n_sentence': 1,
    #         'description': 'protein gene', 'n_description': 1,
    #         'truth': np.arange(D, dtype=np.float32),
    #         'n_truth': D},

    #         {'sentence': 'We believe Biomedical Entity Linking is difficult .', 'n_sentence': 1,
    #         'description': 'mouse protein', 'n_description': 1,
    #         'truth': np.arange(D, dtype=np.float32),
    #         'n_truth': D},

    #         {'sentence': 'We are doing Biomedical Entity Linking and believe it is difficult .', 'n_sentence': 1,
    #         'description': 'mouse protein gene brain', 'n_description': 1,
    #         'truth': np.arange(D, dtype=np.float32),
    #         'n_truth': D}
    #         ]
    #input_sentences = [text for item in data for label, text in item.items() if label in ['sentence', 'description']]
    (vocab_all, vocab_dict, ont_all) = build_consolidatedsentandontmapping(TERMS_FILE, ONT_CONTEXTSFILE, ONTLABELIDMAPPINF_FNAME)
    word_to_idx = build_vocab(vocab_all, threshold=1)
    print(word_to_idx)

    #Generating embedding matrix
    embeddings = get_word_embeddings_from_w2vPMd(EMBEDDING_FNAME,w_idx_map=word_to_idx)

    #Store pre-trained embeddings for each of the ont terms
    for x in vocab_dict:
        sent_tokens = x['sentence'].split(' ')
        sent_array = []
        for st in sent_tokens:
            sent_array.append(word_to_idx.get(st, word_to_idx[data_constants.UNK_TOKEN]))
        x['sentence'] = np.asarray(sent_array)

        desc_array = []
        desc_tokens = x['description'].split(' ')
        for d in desc_tokens:
            desc_array.append(word_to_idx.get(d, word_to_idx[data_constants.UNK_TOKEN]))
        x['description'] = np.asarray(desc_array)
    vocab_size = len(vocab_dict)
    print("Records ", vocab_size)

    #Sort the ontologies by the truth id; so that it is easier to load them in
    ont_all.sort(key = lambda x : x[1])
    print(" Ont peek ",ont_all[:10])
    #Post sorting, store only the 0th terms
    ont_refined = [ont_obj[0] for ont_obj in ont_all]
    '''
    Inefficient, but find a way to add to onts above
    '''
    for ont_term in ont_refined:        
        ont_desc_array = []
        ont_desc_tokens = ont_term['description'].split(' ')
        #Append index of the ont description
        for ont_d in ont_desc_tokens:
            ont_desc_array.append(word_to_idx.get(ont_d, word_to_idx[data_constants.UNK_TOKEN]))
        ont_term['description'] = np.asarray(ont_desc_array)

    #Perform test and train split here, pick like 15% 
    #Shuffling the data for the train and test, and writing them to separate files.
    len_batch = vocab_size
    len_testbatch = int(0.15*len_batch)
    test_portion =np.random.choice(len_batch, len_testbatch,replace=False)
    train_left = list(set([j for j in range(len_batch)]) - set(test_portion))
    train_set = list(map(lambda x:vocab_dict[x],train_left))
    test_set = list(map(lambda x:vocab_dict[x],test_portion))
    #Add ability to build ont file here
    print(" Train ", len(train_left), " Test ",len(test_portion))
    print("Len of ONTS_DESCS ", len(ont_refined))
    print("Sample test sizes ", train_set[0]['sentence'].shape,  "Sent len ", train_set[0]['n_sentence'], " Ont ", train_set[0]['description'].shape, "Ont len ", train_set[0]['n_description'],  " Truth ", train_set[0]['truth'].shape, "Truth len ",train_set[0]['n_truth'] )

    write_tf_records(split_examples=train_set, split_name='train', out_dir='../dumps', n_files=2,
                     feat_key_type=data_constants.NP_FEAT_KEY_TYPE_MAP, rando=True)

    write_tf_records(split_examples=test_set, split_name='test', out_dir='../dumps', n_files=1,
                     feat_key_type=data_constants.NP_FEAT_KEY_TYPE_MAP, rando=True)

    write_tf_records(split_examples=ont_refined, split_name='ontologyall', out_dir='../dumps', n_files=1,
                     feat_key_type=data_constants.NP_ONT_KEY_TYPE_MAP, rando=False)
