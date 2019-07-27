#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The purpose of this little script is to compute the needed datasets for the training, the validation and the test
of the neural networks.
"""

__author__ = "Taillandier Valentin"
__copyright__ = "Copyright (C) 2019, Taillandier Valentin"
__license__ = "GPL"
__version__ = "1.0"

import argparse
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors
try:
    # If progressbar module is available, it will be used to show some progressbars.
    # To install it -> pip3 install progressbar2 --user
    from progressbar import progressbar
except ImportError:
    def progressbar(x):
        return x






def drawMatrix(matrix, x_axis=None, y_axis=None):
    """
    Draw an alignment matrix with given string as axis scale.
    The drawn matrix is stored in the default pyplot container.
    Do not forget to plt.show() if you want to show the matrix
    or to do plt.savefig("mymatrix.pdf") if you want to save the figure.

    This was made for a debug purpose and to make illustration for presentations.

    :param matrix: Numpy 2D-Array, Alignment matrix to show.
    :param x_axis: List of words for the x-axis label.
    :param y_axis: List of words for the y-axis label.
    :return: None
    """
    plt.matshow(matrix, cmap="gray_r")
    if x_axis is not None and y_axis is not None:
        plt.xticks(range(len(y_axis)), y_axis)
        plt.yticks(range(len(x_axis)), x_axis)


def costodistance(x):
    """
    Efficiently computes the Euclidian distance between unit vectors (i.e. L2 norm equal to 1),
    based on their scalar product (i.e. the cosinus of the angle between them).

    If a Numpy ndArray is given, applies efficiently the transformation to every cells.

    :param x: Numpy ndArray like (or a scalar).
    :return: A Numpy ndArray of the same input shape (or a scalar).
    """
    return np.sqrt(2*(1-x))


def resizeSentence(sentence, targetsize, fill=False):
    """
    Takes a string representing a sentence (token separated by spaces) and transforms it so that the number of
    tokens becomes targetsize.

    If fill=False, just pads the sentence with empty tokens (empty string).
    If fill=True, repeats each tokens the same time so that the number of tokens is close to targetsize,
    then pads with #.

    :param sentence: A string, the sentence to resize (tokens separated with spaces).
    :param targetsize: An integer, the target number of tokens.
    :param fill: A boolean (See the function description).
    :return: A string, the resized sentence.
    """

    sentence_tokens = sentence.split(" ")  # Split the sentence into tokens list
    new_sentence_words = list()
    if fill:
        q = targetsize // len(sentence_tokens)  # Computes how many times each tokens will be repeated.
        for word in sentence_tokens:
            new_sentence_words.extend(q*[word])  # Makes the list of repeated tokens.
        new_sentence_words.extend((targetsize - len(new_sentence_words)) * ["#"])  # Pads with '#'
    else:
        new_sentence_words = sentence_tokens
        new_sentence_words.extend((targetsize - len(new_sentence_words)) * ["#"])  # Pads with ''

    sentence = " ".join(new_sentence_words)

    return sentence

def reshapeMatrix(matrix, targetsize, fill=False):
    """
    Takes a 2D Array representing an alignment matrix and transforms it so that its shape becomes
    (targetsize x targetsize).

    If fill=False, just encapsulates the input matrix to a bigger one (add empty rows and columns at the end).
    If fill=True, repeats each cells the same time so that the number of cells of each axis is close to
    targetsize and then pads the bottom right part with 1.

    This is related to resizeSentence and do the same transformation on matrices.


    Take a matrix and expends it to the size x size size.
    Use the "repeat words, then padding" strategy.
    :param matrix: input ndarray like matrix.
    :param targetsize: size of the edges of the output matrix.
    :param fill: A boolean (See the function description).
    :return: a size x size array.
    """

    output_matrix = np.zeros((targetsize, targetsize))  # Initializes the output matrix.
    n, m = matrix.shape

    if fill:
        qn, rn = divmod(targetsize, n)  # Computes how many times too repeat cells for x-axis.
        qm, rm = divmod(targetsize, m)  # Computes how many times too repeat cells for y-axis.
        #  Repeat cells
        ones_matrix = np.ones((qn, qm))
        for i in range(n):
            for j in range(m):
                value = matrix[i, j]
                output_matrix[qn * i:qn * (i + 1), qm * j:qm * (j + 1)] = value * ones_matrix
        #  Add padding
        output_matrix[n * qn:targetsize, m * qm:targetsize] = np.ones((rn, rm))
    else:
        output_matrix[:n, :m] = matrix

    return output_matrix



def monolingualMatrices(sentence_pairs, model_path, dict_to_update, distance_mode=False):
    """
    Builds monolingual alignment matrices using cosine similarities given by a word embedding model.

    :param sentence_pairs: A list of pairs of strings representing sentences, each pair becomes a matrix.
    :param model_path: Path to the word embedding model to use.
        Should be compatible with the Gensim's KeyedVectors.load() method.
    :param dict_to_update: The matrices are stored in the this dictionary (pointer passing),
        the keys are the sentences pairs.
    :param distance_mode: If True, converts the cosine values into distance.
    :return: None (the dictionary is updated with side effect).
    """

    wv_model = KeyedVectors.load(model_path)  # Loads the word embedding model.
    for sentence1, sentence2 in progressbar(sentence_pairs):
        s1_words = sentence1.lower().split(" ")
        s2_words = sentence2.lower().split(" ")
        matrix = None
        try:
            s1_vectors = np.array([wv_model.get_vector(word) for word in s1_words])
            s2_vectors = np.array([wv_model.get_vector(word) for word in s2_words])
            s1_vectors_normalized = s1_vectors/np.linalg.norm(s1_vectors, axis=1, keepdims=True)
            s2_vectors_normalized = s2_vectors/np.linalg.norm(s2_vectors, axis=1, keepdims=True)
            # Computes efficiently all cosines using matrices dot product.
            matrix = s1_vectors_normalized.dot(s2_vectors_normalized.T)
            # Makes sure that the values are in [-1, 1] and avoid float precision errors.
            matrix = np.minimum(np.maximum(matrix, -1), 1)
        except:
            pass

        if matrix is not None:
            assert (np.all(matrix <= 1.1) and np.all(matrix >= -1.1))
            assert (matrix.shape[0] == len(s1_words))
            assert (matrix.shape[1] == len(s2_words))
            if distance_mode:
                matrix = costodistance(matrix)
            dict_to_update[(sentence1, sentence2)] = matrix
            # Also add the reverse pair (for data augmentation)
            dict_to_update[(sentence2, sentence1)] = matrix.T



def bilingualMatrices(sentences_pairs, model_path, dict_to_update):
    """
    Builds bilingual alignment matrices using a bilingual word translation table.
    The word translation table should be store in a text file with this format:
    language1_word language2_word probability

        Example:
        ? ? 0.999888
        may choisir 0.0410224
        may utiliser 0.1287662

    :param sentence_pairs: A list of pairs of strings representing sentences, each pair becomes a matrix.
        The first element of the pair is a sentence in the first language,
        the second element is a sentence in the second language.
    :param model_path: Path to the word embedding model to use.
        Should be compatible with the Gensim's KeyedVectors.load() method.
    :param dict_to_update: The matrices are stored in the this dictionary (pointer passing),
        the keys are the sentences pairs.
    :return: None (the dictionary is updated with side effect).
    """

    #  Loads the bilingual word translation table in a Python dictionary
    translation_table = dict()
    model_file = open(model_path)
    for line in model_file:
        w1, w2, score = line.rstrip('\n').split(" ")
        translation_table[(w1, w2)] = np.float32(score)
    model_file.close()

    # Proceeds the sentences.
    for sentence1, sentence2 in progressbar(sentences_pairs):
        s1_words = sentence1.lower().split(" ")
        s2_words = sentence2.lower().split(" ")

        #  Initiates the matrix
        n = len(s1_words)
        m = len(s2_words)
        matrix = np.zeros((n, m), dtype=np.float32)

        # Fills the matrix.
        for i in range(n):
            for j in range(m):
                try:
                    # Try to find this pair of words in the model
                    matrix[i, j] = translation_table[(s1_words[i], s2_words[j])]
                except:
                    # If the pair of words is not found in the model, the words are not related.
                    # The translation probability is 0.
                    matrix[i, j] = 0

        #  Updates the dictionary with side effect.
        dict_to_update[(sentence1, sentence2)] = matrix


def datasetfromcuboids(cuboids, first_model_path, second_model_path, bilingual_model_path, output_path, fill=False):
    """
    Builds a dataset from the given list of cuboids.
    The dataset is stored to the given path.
    A list of cuboids is a list of pairs of analogies.
    Each analogy is a 4-tuple of string representing sentences.
    The second analogy of a pair is made with the translations of the sentences in the first analogy.

    :param cuboids: List of cuboids.
    :param output_path: Given path to store the data set.
    :param fill: A boolean (See resizeSentence and reshapeMatrix for more details).
    :return: None
    """

    # Initiates the set of pairs of sentences that will be used to produce matrices.
    first_language_pairs = set()
    second_language_pairs = set()
    bilingual_pairs = set()

    # Proceeds all cuboids.
    for analogies_pair in progressbar(cuboids):
        # Extracts the pairs of sentences that will be proceeds to make matrices.
        # For first language monolingual matrices
        first_language_pairs.add((analogies_pair[0][0], analogies_pair[0][1]))
        first_language_pairs.add((analogies_pair[0][2], analogies_pair[0][3]))
        first_language_pairs.add((analogies_pair[0][0], analogies_pair[0][2]))
        first_language_pairs.add((analogies_pair[0][1], analogies_pair[0][3]))
        # For second language monolingual matrices
        second_language_pairs.add((analogies_pair[1][0], analogies_pair[1][1]))
        second_language_pairs.add((analogies_pair[1][2], analogies_pair[1][3]))
        second_language_pairs.add((analogies_pair[1][0], analogies_pair[1][2]))
        second_language_pairs.add((analogies_pair[1][1], analogies_pair[1][3]))
        # For bilingual matrices
        for k in range(4):
            bilingual_pairs.add((analogies_pair[0][k], analogies_pair[1][k]))

    # Initiates the dictionary where all matrices will be stored.
    matrices_dict = dict()

    # Builds the matrices and updates the dictionary with side effect.
    bilingualMatrices(bilingual_pairs, bilingual_model_path, matrices_dict)
    monolingualMatrices(first_language_pairs, first_model_path, matrices_dict)
    monolingualMatrices(second_language_pairs, second_model_path, matrices_dict)
    #  From there, matrices_dict contains all the necessary matrices to build the dataset.

    # For data augmentation, we rotate cuboids along the languages axis.
    # Rotating the cuboids plays with the properties of analogies.
    equivalent_analogies = [
        [0, 1, 2, 3],  # The initial cuboids
        [2, 3, 0, 1],
        [0, 2, 1, 3], [1, 3, 0, 2],
        [1, 0, 3, 2], [3, 2, 1, 0],
        [2, 0, 3, 1], [3, 1, 2, 0]
    ]

    # Computes the index of cuboids
    sentence_pairs, matrices = zip(*matrices_dict.items())
    sentence_pairs_index = {key: index for index, key in enumerate(sentence_pairs)}
    analogy_matrices_list = []
    analogies_lengths_list = []
    for analogies_pair in progressbar(cuboids):
        for i0, i1, i2, i3 in equivalent_analogies:
            try:
                analogy_matrices = [
                    sentence_pairs_index[(analogies_pair[0][i0], analogies_pair[0][i1])],
                    sentence_pairs_index[(analogies_pair[0][i1], analogies_pair[1][i1])],
                    sentence_pairs_index[(analogies_pair[0][i0], analogies_pair[0][i2])],
                    sentence_pairs_index[(analogies_pair[0][i2], analogies_pair[1][i2])],
                    sentence_pairs_index[(analogies_pair[1][i2], analogies_pair[1][i3])],
                    sentence_pairs_index[(analogies_pair[1][i1], analogies_pair[1][i3])],

                    sentence_pairs_index[(analogies_pair[0][i0], analogies_pair[1][i0])],
                    sentence_pairs_index[(analogies_pair[1][i0], analogies_pair[1][i2])],
                    sentence_pairs_index[(analogies_pair[1][i0], analogies_pair[1][i1])],
                ]
                analogy_matrices_list.append(analogy_matrices)

                la, lb = matrices[analogy_matrices[0]].shape
                lc, lc2 = matrices[analogy_matrices[3]].shape
                la2, lb2 = matrices[analogy_matrices[8]].shape
                _, ld2 = matrices[analogy_matrices[4]].shape

                analogies_lengths_list.append((la, lb, lc, lb2, lc2, ld2, la2))

            except:
                pass

    # Computes the length of the biggest sentence
    biggest_sentence_length = 0
    for lengths in analogies_lengths_list:
        biggest_sentence_length = max(biggest_sentence_length, max(lengths))

    print("Length of the biggest sentence:", biggest_sentence_length)

    # Reshape matrices and resize sentences to the length of the biggest sentence.
    matrices = [reshapeMatrix(value, biggest_sentence_length, fill=fill) for value in matrices]
    sentence_pairs = [
        (resizeSentence(s1, biggest_sentence_length, fill),
         resizeSentence(s1, biggest_sentence_length, fill)) for s1, s2 in sentence_pairs]

    # Stacks all matrices into a big 3D array.
    matrices = np.stack(matrices, axis=0)

    # Save all arrays using the Numpy npz format (simple tar archive of arrays in np storing format)
    np.savez(output_path, index=sentence_pairs, analogies=analogy_matrices_list, matrices=matrices,
             lengths=analogies_lengths_list)



if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--padding', dest='padding', action='store_true',
                        help='use the "Repeat words and add padding" to fulfill the matrix')
    parser.add_argument('--input', dest='input_path', action='store',
                        help='cuboid analogies textfile to proceed.')
    parser.add_argument('--output', dest='output_path', action='store',
                        help='output name to store matrices.')
    parser.add_argument('--first_language_model', dest='first_model_path', action='store',
                        help='word embeding model path. Will be open with gensim.open()')
    parser.add_argument('--second_language_model', dest='second_model_path', action='store',
                        help='word embeding model path. Will be open with gensim.open()')
    parser.add_argument('--bilingual_model', dest='bilingual_model_path', action='store',
                        help='word translation model path. Should looks like a trained Hieraligne model')
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    first_model_path = args.first_model_path
    second_model_path = args.second_model_path
    bilingual_model_path = args.bilingual_model_path
    padding = args.padding


    # Load cuboids text file
    input_file = open(input_path)
    lines = input_file.read().splitlines()
    input_file.close()
    assert(len(lines) % 2 == 0)

    number_of_cuboids = len(lines) // 2
    cuboids_index = list(range(number_of_cuboids))
    # Shuffle at random
    shuffle(cuboids_index)

    train_range = 0, int(0.6 * number_of_cuboids)
    validation_range = train_range[1], int(0.8 * number_of_cuboids)
    test_range = validation_range[1], number_of_cuboids

    cuboids = [
        (lines[2 * cuboids_index[i]].split("\t"),
         lines[2 * cuboids_index[i] + 1].split("\t"))
        for i in range(number_of_cuboids)]


    train_cuboids = cuboids[train_range[0]:train_range[1]]
    validation_cuboids = cuboids[validation_range[0]:validation_range[1]]
    test_cuboids = cuboids[test_range[0]:test_range[1]]

    # Makes all datasets
    datasetfromcuboids(train_cuboids, first_model_path, second_model_path, bilingual_model_path,
                       output_path + ".train", fill=padding)
    datasetfromcuboids(validation_cuboids, first_model_path, second_model_path, bilingual_model_path,
                       output_path + ".validation", fill=padding)
    datasetfromcuboids(test_cuboids, first_model_path, second_model_path, bilingual_model_path,
                       output_path + ".test", fill=padding)








