#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import glob
import pickle as pickle
from gensim.models import KeyedVectors, keyedvectors
from gensim.models import fasttext
from gensim.models import word2vec
from shutil import copyfile
from os import remove
from os.path import realpath
from os import stat
import csv
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt

import argparse
from progressbar import progressbar



def cossim2distance(x):
    """

    :param x: a scalar value or a np.array like (in this case, apply elementwise).
    :return: The distance base on x seen as the cosine similarity (return sqrt(2-2x))
    """
    # distance = sqrt(2(1-cossim(a,b)))
    return np.sqrt(2*(1-np.minimum(x, 1)))





def monolingualMatrices(sentences_couples, model_path, dict_to_update):
    wv_model = KeyedVectors.load(model_path)
    for sentence1, sentence2 in progressbar(sentences_couples):
        s1_words = word_tokenize(sentence1.lower())
        s2_words = word_tokenize(sentence2.lower())
        try:
            s1_vectors = np.array([wv_model.get_vector(word) for word in s1_words])
            s2_vectors = np.array([wv_model.get_vector(word) for word in s2_words])
            s1_vectors_normalized = s1_vectors/np.linalg.norm(s1_vectors, axis=1, keepdims=True)
            s2_vectors_normalized = s2_vectors/np.linalg.norm(s2_vectors, axis=1, keepdims=True)
            cossim_matrix = s1_vectors_normalized.dot(s2_vectors_normalized.T)
            distance_matrix = cossim2distance(cossim_matrix)
            dict_to_update[(sentence1, sentence2)] = distance_matrix
        except:
            pass

def bilingualMatrices(sentences_couples, model_path, dict_to_update):
    tt_model = dict()
    model_file = open(model_path)
    for line in model_file:
        w2, w1, score = line.rstrip('\n').split(" ")
        #w1, w2, score = line.rstrip('\n').split(" ")
        tt_model[(w1, w2)] = np.float32(score)
    model_file.close()


    for sentence1, sentence2 in progressbar(sentences_couples):
        s1_words = word_tokenize(sentence1.lower())
        s2_words = word_tokenize(sentence2.lower())
        n = len(s1_words)
        m = len(s2_words)
        matrix = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            for j in range(m):
                try:
                    matrix[i, j] = tt_model[(s1_words[i], s2_words[j])]
                except:
                    matrix[i, j] = 0
        ponderation = np.sqrt(np.sum(matrix, axis=1, keepdims=True).dot(np.sum(matrix, axis=0, keepdims=True)))

        matrix = np.divide(matrix, ponderation, where=ponderation != 0)
        dict_to_update[(sentence1, sentence2)] = matrix


def couple_sort(couple):
    if couple[0] <= couple[1]:
        return couple
    else:
        return couple[1], couple[0]


if __name__ == '__main__':

    #Argz parsing
    parser = argparse.ArgumentParser(description='Computes alignment matrices.')
    parser.add_argument('--input', dest='input_path', action='store', help='Cuboid analogies textfile to proceed.')
    parser.add_argument('--output', dest='output_path', action='store', help='Output name to store matrices.')
    parser.add_argument('--first_language_model', dest='model1_path', action='store', help='Word embeding model path. Will be open with gensim.open().')
    parser.add_argument('--second_language_model', dest='model2_path', action='store', help='Word embeding model path. Will be open with gensim.open().')
    parser.add_argument('--bilingual_model', dest='bilingual_model_path', action='store', help='Word translation model path. Should looks like a trained Hieraligne model.')
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    model1_path = args.model1_path
    model2_path = args.model2_path
    bilingual_model_path = args.bilingual_model_path


    #Load sentences
    input_file = open(input_path)
    lines = input_file.read().splitlines()
    input_file.close()
    assert(len(lines) % 2 == 0)  # One analogie tooks 2 lines. So there is an even number of lines.
    language1_couples = set()
    language2_couples = set()
    bilingual_couples = set()
    for i in progressbar(range(0, 10, 2)):
        sentences = [lines[i].split("\t"), lines[i+1].split("\t")]
        language1_couples.add(couple_sort((sentences[0][0], sentences[0][1])))
        language1_couples.add(couple_sort((sentences[0][2], sentences[0][3])))
        language1_couples.add(couple_sort((sentences[0][0], sentences[0][2])))
        language1_couples.add(couple_sort((sentences[0][1], sentences[0][3])))

        language2_couples.add(couple_sort((sentences[1][0], sentences[1][1])))
        language2_couples.add(couple_sort((sentences[1][2], sentences[1][3])))
        language2_couples.add(couple_sort((sentences[1][0], sentences[1][2])))
        language2_couples.add(couple_sort((sentences[1][1], sentences[1][3])))

        bilingual_couples.add((sentences[0][0], sentences[1][0]))
        bilingual_couples.add((sentences[0][1], sentences[1][1]))
        bilingual_couples.add((sentences[0][2], sentences[1][2]))
        bilingual_couples.add((sentences[0][3], sentences[1][3]))

    matrices_dict = dict()
    monolingualMatrices(language1_couples, model1_path, matrices_dict)
    monolingualMatrices(language2_couples, model2_path, matrices_dict)
    bilingualMatrices(bilingual_couples, bilingual_model_path, matrices_dict)

    keys, values = zip(*matrices_dict.items())
    values_dict = dict()
    for i, value in enumerate(values):
        values_dict[str(i)] = value
    np.savez(output_path, index=keys, **values_dict)







