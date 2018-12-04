import numpy as np

STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}


def generate_word_to_vec_map():
    """
    generating map of word to vector
    :return: word to vec map
    """
    word_to_vec_map = {}
    with open('vocab', 'r') as f:
        embedding = np.loadtxt("wordVectors")
        for ind, vocab_line in enumerate(f.readlines()):
            w = vocab_line.strip()
            word_to_vec_map[w] = embedding[ind]
    return word_to_vec_map


def most_similar(word, k):
    """
    calculating k most similar words of the word
    :param word: current word
    :param k: size of the output
    :return: list of k most similar words
    """
    word_to_vec_map = generate_word_to_vec_map()
    if word in word_to_vec_map:
        u = word_to_vec_map[word]
        dot_values = {}
        # calculating the distance for each vector
        for key, v in word_to_vec_map.items():
            # skip same word
            if key == word:
                continue
            dot_values[key] = similarity_measure(u, v)
        # sort the output map by the value in reverse order
        sorted_dot_values = sorted(dot_values, key=dot_values.get, reverse=True)
        return [word for i, word in enumerate(sorted_dot_values) if i < k]
    return []


def similarity_measure(u, v):
    """
    Calculate cosine distance of two vectors.
    :param u: First vector.
    :param v: Second vector.
    :return: Distance value.
    """
    return np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v)))


if __name__ == '__main__':
    print(most_similar('dog', 5))
    print(most_similar('england', 5))
    print(most_similar('john', 5))
    print(most_similar('explode', 5))
    print(most_similar('office', 5))
