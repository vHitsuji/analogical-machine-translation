#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
from progressbar import progressbar
from math import sqrt




if __name__ == '__main__':



    #Argz parsing
    parser = argparse.ArgumentParser(description='Extract bitexts from cuboid analogies file.')
    parser.add_argument('--input_l1_l2', dest='input12_path', action='store',
                        help='Translation model from language1 to language2.')
    parser.add_argument('--input_l2_l1', dest='input21_path', action='store',
                        help='Translation model from language2 to language1')
    parser.add_argument('--output', dest='output_path', action='store',
                        help='Output name for the language1 to language2 symetric model.')
    args = parser.parse_args()
    input12_path = args.input12_path
    input21_path = args.input21_path
    output_path = args.output_path

    #Load sentences
    input12_file = open(input12_path)
    input21_file = open(input21_path)
    lines12 = input12_file.read().splitlines()
    lines21 = input21_file.read().splitlines()
    input12_file.close()
    input21_file.close()

    model12_dict = dict()
    model21_dict = dict()

    for i in range(len(lines12)):
        w1, w2, p = lines12[i].split(" ")
        model12_dict[(w1, w2)] = float(p)


    for i in range(len(lines21)):
        w1, w2, p = lines21[i].split(" ")
        model21_dict[(w1, w2)] = float(p)

    model12_symetric_dict = dict()
    for key, value in model12_dict.items():
        w1, w2 = key
        try:
            model12_symetric_dict[(w1, w2)] = sqrt(value*model21_dict[(w2, w1)])
        except:
            pass


    model12_symetric_string = "\n".join([" ".join((w1, w2, str(round(p, 7)))) for ((w1, w2), p) in model12_symetric_dict.items()])

    output_file = open(output_path, "w+")
    output_file.write(model12_symetric_string)
    output_file.close






