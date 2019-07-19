#!/usr/bin/env python
# -*- coding: utf-8 -*-



from gensim.models import KeyedVectors, keyedvectors
from nltk.tokenize import word_tokenize
import numpy as np
import argparse
from progressbar import progressbar
import matplotlib.pyplot as plt
from random import shuffle
import seaborn as sns
from collections import Counter


from scipy.optimize import linprog


def simplexAlign(matrix):

    n, m = matrix.shape
    all_constraint = list()
    for i in range(n):
        constraint = list()
        for j in range(n):
            for k in range(m):
                if i == j:
                    constraint.append(1)
                else:
                    constraint.append(0)
        all_constraint.append(constraint)

    for i in range(m):
        constraint = list()
        for j in range(n):
            for k in range(m):
                if i == k:
                    constraint.append(1)
                else:
                    constraint.append(0)
        all_constraint.append(constraint)

    all_constraint_rh = np.array([1] * len(all_constraint))
    all_constraint = np.array(all_constraint)
    c = -np.array(matrix.flatten())
    result = linprog(c, A_ub=all_constraint, b_ub=all_constraint_rh).x
    result.shape = n, m
    return result


def printMatrix(matrix, s1_words=None, s2_words=None):
    """
    Print an alignment matrix with given string as axis scale.
    Do not forget to plt.show() when you want to show the matrix.
    :param matrix: Matrix to show.
    :param s1_words: x_axis sentence.
    :param s2_words: y_axis sentence.
    :return: None
    """
    plt.matshow(matrix, cmap="gray_r")
    if s1_words is not None and s1_words is not None:
        plt.xticks(range(len(s2_words)), s2_words)
        plt.yticks(range(len(s1_words)), s1_words)

def cossim2distance(x):
    """

    :param x: a scalar value or a np.array like (in this case, apply elementwise).
    :return: The distance base on x seen as the cosine similarity (return sqrt(2-2x))
    """
    return np.sqrt(2*(1-np.minimum(x, 1)))


def reshapeSentence(sentence, size, padding=False):
    sentence_words = sentence.split(" ")
    new_sentence_words = list()
    if padding:
               q = size//len(sentence_words)
               for word in sentence_words:
                   new_sentence_words.extend(q*[word])


    else:
        new_sentence_words = sentence_words
    new_sentence_words.extend((size - len(new_sentence_words)) * ["#"])
    sentence = " ".join(new_sentence_words)
    return sentence

def reshapeMatrix(matrix, size, padding=False):
    """
    Take a matrix and expend it to the size x size size.
    Use the "repeat words, then padding" strategy.
    :param matrix: input ndarray like matrix.
    :param size: size of the edges of the output matrix.
    :return: a size x size array.
    """

    output_matrix = np.zeros((size, size))
    n, m = matrix.shape

    if padding:
        qn, rn = divmod(size, n)
        qm, rm = divmod(size, m)

        #  Repeat words
        ones_matrix = np.ones((qn, qm))
        for i in range(n):
            for j in range(m):
                value = matrix[i, j]
                output_matrix[qn * i:qn * (i + 1), qm * j:qm * (j + 1)] = value * ones_matrix

        #  Add padding
        output_matrix[n * qn:size, m * qm:size] = np.ones((rn, rm))
    else:
        output_matrix[:n, :m] = matrix

    return output_matrix





def monolingualMatrices(sentences_couples, model_path, dict_to_update):
    wv_model = KeyedVectors.load(model_path)
    #distrib_list = list()
    for sentence1, sentence2 in progressbar(sentences_couples):
        s1_words = sentence1.lower().split(" ")
        s2_words = sentence2.lower().split(" ")
        matrix = None
        try:
            s1_vectors = np.array([wv_model.get_vector(word) for word in s1_words])
            s2_vectors = np.array([wv_model.get_vector(word) for word in s2_words])
            s1_vectors_normalized = s1_vectors/np.linalg.norm(s1_vectors, axis=1, keepdims=True)
            s2_vectors_normalized = s2_vectors/np.linalg.norm(s2_vectors, axis=1, keepdims=True)
            matrix = s1_vectors_normalized.dot(s2_vectors_normalized.T)


        except:
            pass

        if matrix is not None:
            assert (np.all(matrix <= 1.1) and np.all(matrix >= -1.1))

            assert (matrix.shape[0] == len(s1_words))
            assert (matrix.shape[1] == len(s2_words))
            #matrix = cossim2distance(matrix)
            dict_to_update[(sentence1, sentence2)] = matrix
            dict_to_update[(sentence2, sentence1)] = matrix.T




            #distrib_list.extend(matrix.flatten().tolist())


    #sns.distplot(distrib_list[:10000]+distrib_list[-10000:])
    #plt.xlabel("Values")
    #plt.ylabel("Density")
    #plt.show()

def bilingualMatrices(sentences_couples, model_path, dict_to_update):
    #distrib_list = list()
    tt_model = dict()
    model_file = open(model_path)
    for line in model_file:
        w1, w2, score = line.rstrip('\n').split(" ")
        tt_model[(w1, w2)] = np.float32(score)
    model_file.close()


    for sentence1, sentence2 in progressbar(sentences_couples):
        s1_words = sentence1.lower().split(" ")
        s2_words = sentence2.lower().split(" ")
        n = len(s1_words)
        m = len(s2_words)
        matrix = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            for j in range(m):
                try:
                    matrix[i, j] = tt_model[(s1_words[i], s2_words[j])]
                except:
                    matrix[i, j] = 0

        assert(np.all(matrix <= 1) and np.all(matrix >= -1))
        dict_to_update[(sentence1, sentence2)] = matrix

        #distrib_list.extend(matrix.flatten().tolist())


    #sns.distplot(distrib_list[:10000]+distrib_list[-10000:])
    #plt.xlabel("Values")
    #plt.ylabel("Density")
    #plt.show()


def couple_sort(couple):
    if couple[0] <= couple[1]:
        return couple
    else:
        return couple[1], couple[0]


def proceed(cuboids, filename):

    language1_couples = set()
    language2_couples = set()
    bilingual_couples = set()

    for analogy in progressbar(cuboids):
        language1_couples.add((analogy[0][0], analogy[0][1]))
        language1_couples.add((analogy[0][2], analogy[0][3]))
        language1_couples.add((analogy[0][0], analogy[0][2]))
        language1_couples.add((analogy[0][1], analogy[0][3]))

        language2_couples.add((analogy[1][0], analogy[1][1]))
        language2_couples.add((analogy[1][2], analogy[1][3]))
        language2_couples.add((analogy[1][0], analogy[1][2]))
        language2_couples.add((analogy[1][1], analogy[1][3]))

        for k in range(4):
            bilingual_couples.add((analogy[0][k], analogy[1][k]))

    matrices_dict = dict()

    bilingualMatrices(bilingual_couples, bilingual_model_path, matrices_dict)
    monolingualMatrices(language1_couples, model1_path, matrices_dict)
    monolingualMatrices(language2_couples, model2_path, matrices_dict)

    #  From there, matrices_dict contains all the necessary matrices
    equivalent_analogies = [
        [0, 1, 2, 3], [2, 3, 0, 1],
        [0, 2, 1, 3], [1, 3, 0, 2],
        [1, 0, 3, 2], [3, 2, 1, 0],
        [2, 0, 3, 1], [3, 1, 2, 0]
    ]

    sentence_couples, values = zip(*matrices_dict.items())
    keys_index = {key: index for index, key in enumerate(sentence_couples)}

    analogy_matrices_list = []
    analogies_lengths_list = []
    for analogy in progressbar(cuboids):
        for i0, i1, i2, i3 in equivalent_analogies:
            try:
                analogy_matrices = [
                    keys_index[(analogy[0][i0], analogy[0][i1])],
                    keys_index[(analogy[0][i1], analogy[1][i1])],
                    keys_index[(analogy[0][i0], analogy[0][i2])],
                    keys_index[(analogy[0][i2], analogy[1][i2])],
                    keys_index[(analogy[1][i2], analogy[1][i3])],
                    keys_index[(analogy[1][i1], analogy[1][i3])],

                    keys_index[(analogy[0][i0], analogy[1][i0])],
                    keys_index[(analogy[1][i0], analogy[1][i2])],
                    keys_index[(analogy[1][i0], analogy[1][i1])],
                ]
                analogy_matrices_list.append(analogy_matrices)

                la, lb = values[analogy_matrices[0]].shape
                lc, lc2 = values[analogy_matrices[3]].shape
                la2, lb2 = values[analogy_matrices[8]].shape
                _, ld2 = values[analogy_matrices[4]].shape

                analogies_lengths_list.append((la, lb, lc, lb2, lc2, ld2, la2))

            except:
                pass

    biggest_sentence_length = 0
    for lengths in analogies_lengths_list:
        biggest_sentence_length = max(biggest_sentence_length, max(lengths))

    print("Length of the biggest sentence:", biggest_sentence_length)

    matrices = [reshapeMatrix(value, biggest_sentence_length, padding=padding) for _, value in enumerate(values)]
    sentence_couples = [(reshapeSentence(s1, biggest_sentence_length, padding),
                         reshapeSentence(s1, biggest_sentence_length, padding)) for s1, s2 in sentence_couples]
    matrices = np.stack(matrices, axis=0)
    np.savez(output_path + "." + filename, index=sentence_couples, analogies=analogy_matrices_list, matrices=matrices,
             lengths=analogies_lengths_list)


if __name__ == '__main__':

    #Argz parsing
    parser = argparse.ArgumentParser(description='Computes alignment matrices.')
    parser.add_argument('--padding', dest='padding', action='store_true',
                        help='Use the "Repeat words and add padding" to fulfill the matrix.')
    parser.add_argument('--input', dest='input_path', action='store',
                        help='Cuboid analogies textfile to proceed.')
    parser.add_argument('--output', dest='output_path', action='store',
                        help='Output name to store matrices.')
    parser.add_argument('--first_language_model', dest='model1_path', action='store',
                        help='Word embeding model path. Will be open with gensim.open().')
    parser.add_argument('--second_language_model', dest='model2_path', action='store',
                        help='Word embeding model path. Will be open with gensim.open().')
    parser.add_argument('--bilingual_model', dest='bilingual_model_path', action='store',
                        help='Word translation model path. Should looks like a trained Hieraligne model.')
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    model1_path = args.model1_path
    model2_path = args.model2_path
    padding = args.padding
    bilingual_model_path = args.bilingual_model_path



    #Load sentences
    input_file = open(input_path)
    lines = input_file.read().splitlines()
    input_file.close()
    assert(len(lines) % 2 == 0)  # One analogy takes 2 lines. So there is an even number of lines.

    bilines_number = len(lines)//2
    bilines_index = list(range(bilines_number))
    shuffle(bilines_index)
    train_number = int(0.8*bilines_number)
    test_number = bilines_number - train_number

    train_cuboids = list()
    test_cuboids = list()
    for i in range(train_number):
        line1 = lines[2*bilines_index[i]]
        line2 = lines[2*bilines_index[i] + 1]
        train_cuboids.append([line1.split("\t"), line2.split("\t")])
    for i in range(train_number, bilines_number):
        line1 = lines[2*bilines_index[i]]
        line2 = lines[2*bilines_index[i] + 1]
        test_cuboids.append([line1.split("\t"), line2.split("\t")])




    proceed(train_cuboids, "train")
    proceed(test_cuboids, "test")









