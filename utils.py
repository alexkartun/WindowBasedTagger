import numpy as np

STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}

# util's globals
START_SYMBOL = '^START'
STOP_SYBMOL = '^END'
UNK_SYMBOL = 'UUUNKKK'
words_set = {UNK_SYMBOL}
tags_set = set()
# util's maps
W2I = {}
I2W = {}
T2I = {}
I2T = {}
# extra util's maps
P2I = {}
I2P = {}
S2I = {}
I2S = {}
sub_word_units_size = 3


def generate_embedding_vecs():
    """
    extracting word embedded vectors and converting to numpy arrays
    :return: numpy arrays of embedded vectors
    """
    return np.loadtxt("wordVectors")


def generate_words_set():
    """
    generating words set from given vocabulary
    :return:
    """
    with open('vocab', 'r') as f:
        for ind, vocab_line in enumerate(f.readlines()):
            w = vocab_line.strip()
            words_set.add(w)


def generate_tags_set(filename):
    """
    generating tags set from train data set, skipping empty lines
    :param filename:
    :return:
    """
    with open(filename, 'r') as f:
        for line in f.readlines():
            if not line.strip():
                continue
            tag = line.strip().split()[1]
            tags_set.add(tag)


def generate_util_sets(filename):
    """
    generating words set and tags set from train data set
    :param filename: file name of train data set
    :return:
    """
    with open(filename, 'r') as f:
        for line in f.readlines():
            if not line.strip():
                continue
            word, tag = line.strip().split()
            words_set.add(word)
            tags_set.add(tag)


def generete_extra_util_maps():
    """
    generate extra maps for sub word units task, calculating all prefixes and suffixes of length of 3
    and from them generating prefix to index, suffix to index and vice versa maps
    :return:
    """
    prefix_set = {w[:sub_word_units_size] for w in words_set}
    suffix_set = {w[-sub_word_units_size:] for w in words_set}
    P2I.update({p: i for i, p in enumerate(prefix_set)})
    I2P.update({i: p for p, i in P2I.items()})
    S2I.update({s: i for i, s in enumerate(suffix_set)})
    I2S.update({i: s for s, i in S2I.items()})


def generate_util_maps():
    """
    generating word to index, tag to index and vice versa maps from already calculated word and tag sets
    :return:
    """
    W2I.update({w: i for i, w in enumerate(words_set)})
    I2W.update({i: w for w, i in W2I.items()})
    T2I.update({t: i for i, t in enumerate(tags_set)})
    I2T.update({i: t for t, i in T2I.items()})


def generate_validation_data(filename):
    """
    generate feature vectors of test/validation data set
    :param filename: file name of test data set
    :return: feature vectors
    """
    sentences = extract_validation_sentences(filename)
    features_vecs = generate_validation_features_vecs(sentences)
    return features_vecs


def generate_input_data(filename):
    """
    generate feature and target vectors of train/dev data
    :param filename: file name of train/dev data set
    :return: feature and target vectors
    """
    sentences = extract_input_sentences(filename)
    targets_vecs = generate_targets_vecs(sentences)
    features_vecs = generate_input_features_vecs(sentences)
    return features_vecs, targets_vecs


def extract_validation_sentences(filename):
    """
    generating sentences of words, in addition for each sentence adding two strings of start symbol in the beginning
    and two strings of end symbol in the end, skipping empty lines
    :param filename: file name of test/validation data sets
    :return: list of sentences
    """
    sentences = []
    sentence = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if not line.strip():
                sentences.append([START_SYMBOL] * 2 + sentence + [STOP_SYBMOL] * 2)
                sentence = []
                continue
            word = line.strip()
            sentence.append(word)
    return sentences


def extract_input_sentences(filename):
    """
    generating sentences where each sentence is list of tuples, where which tuple is in format of (word, tag)
    in addition for each sentence adding two tuples of start symbol in the beginning and two tuples
    of end symbol in the end, skipping empty lines
    :param filename: file name of train/dev data sets
    :return: list of sentences
    """
    sentences = []
    sentence = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if not line.strip():
                sentences.append([(START_SYMBOL, START_SYMBOL)] * 2 + sentence + [(STOP_SYBMOL, STOP_SYBMOL)] * 2)
                sentence = []
                continue
            word, tag = line.strip().split()
            sentence.append((word, tag))
    return sentences


def generate_targets_vecs(sentences):
    """
    generating target vectors from already calculated sentences where each sentence is list of tuples (word, tag)
    :param sentences: list of sentences
    :return: list of target vectors
    """
    targets_vecs = []
    for sentence in sentences:
        for i, (word, tag) in enumerate(sentence[2:-2], 2):
            targets_vecs.append(T2I[tag])
    return targets_vecs


def generate_validation_features_vecs(sentences):
    """
    generating feature vectors of test/validation data set in window-based pattern, window of 5 words for each word,
    two to the left, current word, and two to the right. skipping first two and last two embedded words
    :param sentences: list of sentences where each sentence is list of words with embedded two start and two end
    symbols in the beginning and the end of sentence respectively
    :return: list of feature vectors
    """
    features = []
    for sentence in sentences:
        for i, word in enumerate(sentence[2:-2], 2):
            features.append(get_word_window(sentence[i - 2], sentence[i - 1], word, sentence[i + 1],
                                            sentence[i + 2]))
    return features


def generate_input_features_vecs(sentences):
    """
    generating feature vectors of train/dev data set in window-based pattern, window of 5 words for each word,
    two to the left, current word, and two to the right. skipping first two and last two embedded tuples
    :param sentences: list of sentences where each sentence is list of tuples with embedded two start tuples
    and two end tuples in the beginning and the end of sentence respectively
    :return: list of feature vectors
    """
    features = []
    for sentence in sentences:
        for i, (word, tag) in enumerate(sentence[2:-2], 2):
            features.append(get_word_window(sentence[i - 2][0], sentence[i - 1][0], word, sentence[i + 1][0],
                                            sentence[i + 2][0]))
    return features


def get_word_window(pp_w, p_w, w, n_w, nn_w):
    """
    getting list of indexes for each word in the window
    :param pp_w: previous of previous word
    :param p_w: previous word
    :param w: current word
    :param n_w: next word
    :param nn_w: next of next word
    :return: list of indexes
    """
    return [get_word_index(pp_w), get_word_index(p_w), get_word_index(w), get_word_index(n_w),
            get_word_index(nn_w)]


def get_word_index(w):
    """
    getting index of word in the util word to index map, if the word is in the vocabulary so return the index,
    else if the lower-case of the word in the vocabulary so return the index of lower-case word, otherwise return
    the index of UNK symbol
    :param w: current word to map
    :return: calculated index of the word
    """
    if w in words_set:
        return W2I[w]
    elif w.lower() in words_set:
        return W2I[w.lower()]
    return W2I[UNK_SYMBOL]
