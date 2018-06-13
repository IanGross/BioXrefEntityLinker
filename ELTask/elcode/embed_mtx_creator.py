import pickle
from collections import Counter

import numpy as np
import re
from more_itertools import unique_everseen
import nltk
import gensim 
import sys
import json

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
def _build_vocab(annotations, threshold=1):
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

    word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    idx = len(word_to_idx)
    for word in vocab_words:
        word_to_idx[word] = idx
        idx += 1
    print("Max length of caption: %d" % max_len)
    return word_to_idx


# If live, then w_idx_map should be the already loaded word_to_idx dictionary
# https://gist.github.com/ottokart/673d82402ad44e69df85
# https://groups.google.com/forum/#!topic/word2vec-toolkit/GFNZkoDPd0g
def get_word_embeddings_from_binary(embedding_fname, init_range=(-0.05, 0.05), w_idx_map='./',
                                    outfname='', verbose=True, live=False, embed_dim=300):
    """
        Loads 300x1 word vecs from Google (Mikolov) word2vec.
        Words in w_idx_map that does not have embeddings in word2vec embeddings are given random embeddings
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
        header = ""
        while c != "\n":
            c = f.read(1)
            header += c

        total_num_vectors, vector_len = (int(x) for x in header.split())
        if verbose:
            print("Number of vectors in embedding file: %d" % total_num_vectors)
            print("Vector size: %d" % vector_len)
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

'''
Function should read in both sent and ont consolidated sentences and build a common vocabulary list
And a list of tuple data_pairs
Files: ../dumps/word_contexts.pkl, ../dumps/unique_truthconts.pkl
'''
def build_consolidatedsentandontmapping(terms_file, ont_contextsfile, labelmapping_file):
    vocab_all = set()
    #List of tuples, sent; ont
    vocab_dict = []

    labelidmapping = load_json(labelmapping_file)
    num_truths = len(labelidmapping)
    terms_cont = load_pickle(terms_file)
    ont_cont = load_pickle(ont_contextsfile)
    for term_neighborhood in terms_cont:
        sent_consolidated = sent_preprocessingandcleanup(term_neighborhood['consolidated'])
        associatedont_consolidated = sent_preprocessingandcleanup(ont_cont[term_neighborhood['truth']]['consolidated'])
        truth_id = int(labelidmapping[term_neighborhood['truth']])
        truth_np =np.zeros(shape=(num_truths,1),dtype=np.float32)
        truth_np[truth_id - 1] = 1
        vocab_all.add(sent_consolidated)
        vocab_all.add(associatedont_consolidated)
        vocab_dict.append((sent_consolidated,associatedont_consolidated, truth_np))

    return (list(vocab_all), vocab_dict)


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
# FILL IN CODE BELOW
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


if __name__ == '__main__':
    (vocab_all, vocab_dict) = build_consolidatedsentandontmapping(TERMS_FILE, ONT_CONTEXTSFILE, ONTLABELIDMAPPINF_FNAME)
    # input_sentences = ['Hello , my name is Spencer .',
    #                    'We are doing Biomedical Entity Linking .',
    #                    'We believe Biomedical Entity Linking is difficult .',
    #                    'mouse mouse mouse',
    #                    'protein protein']
    word_to_idx = _build_vocab(vocab_all, threshold=1)
    print(word_to_idx)    

    embeddings = get_word_embeddings_from_w2vPMd(EMBEDDING_FNAME,w_idx_map=word_to_idx)
    # protein = embeddings[word_to_idx['protein'], :] ==> this will return the embedding of 'protein'

    # data_pairs = [('Hello , my name is Spencer .', 'We is mouse'),
    #               ('We are doing Biomedical Entity Linking .', 'protein'),
    #               ('We believe Biomedical Entity Linking is difficult .', 'mouse')]
    idx_vec_pairs = []
    for sent, desc, truth_np in vocab_dict:
        sent_tokens = sent.split(' ')
        sent_array = []
        for st in sent_tokens:
            sent_array.append(word_to_idx.get(st, word_to_idx[UNK_TOKEN]))

        desc_array = []
        desc_tokens = desc.split(' ')
        for d in desc_tokens:
            desc_array.append(word_to_idx.get(d, word_to_idx[UNK_TOKEN]))

        idx_vec_pairs.append((np.asarray(sent_array), np.asarray(desc_array), truth_np))
    print("Number of pairs ",len(idx_vec_pairs))


