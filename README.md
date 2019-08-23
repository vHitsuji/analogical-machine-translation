
# Merging the direct and indirect approaches in machine translation by analogy.

This repository contains the source code for the design of neural network architectures and the creation of training, validation and test sets.
For more information about this work, visit [http://valentin-taillandier.fr/projects/merging_approaches.pdf](http://valentin-taillandier.fr/projects/merging_approaches.pdf).

### Prerequisites
The list of required python libraries is given below:

```
matplotlib
numpy
pytorch
seaborn
nltk
gensim
```
## Description of files
 ### `makeparalleltext.py`
 This script aims to extract a parallel text from a cuboids (octuplets) file text.
 The resulted parallel text can be used to train an alignment software in order to build the words translations tables. 
 

    usage: makeparalleltext.py [-h] [--input INPUT_PATH]
                           [--output_l1_l2 FIRST_TO_SECOND_PATH]
                           [--output_l2_l1 SECOND_TO_FIRST_PATH]

    The purpose of this little script is to extract a parallel corpus from a
    cuboids corpus.

    optional arguments:
    -h, --help            show this help message and exit
    --input INPUT_PATH    Cuboids text file path.
    --output_l1_l2 FIRST_TO_SECOND_PATH
                        Output path for the first language to second language
                        parallel corpus.
    --output_l2_l1 SECOND_TO_FIRST_PATH
                        Output path for the second language to first language
                        parallel corpus.


 ### `symmetrizeBilingualModel.py`
 This script aims to combine two translations tables (from two languages and the two directions) into a symmetric translation table. This is necessary to build symmetric matrices later.
 

    usage: symmetrizeBilingualModel.py [-h] [--input_l1_l2 INPUT12_PATH]
                                       [--input_l2_l1 INPUT21_PATH]
                                       [--output OUTPUT_PATH]
    
    Extract bitexts from cuboid analogies file.
    
    optional arguments:
      -h, --help            show this help message and exit
      --input_l1_l2 INPUT12_PATH
                            Translation model from language1 to language2.
      --input_l2_l1 INPUT21_PATH
                            Translation model from language2 to language1
      --output OUTPUT_PATH  Output name for the language1 to language2 symetric
                            model.

 
 
 ### `makedataset.py`
This script aims to split at random the cuboids (octuplets) into three parts (60%, 20%, 20%) and computes the alignment matrices in order to build the training, validation and test sets.

    usage: makedataset.py [-h] [--padding] [--input INPUT_PATH]
                          [--output OUTPUT_PATH]
                          [--first_language_model FIRST_MODEL_PATH]
                          [--second_language_model SECOND_MODEL_PATH]
                          [--bilingual_model BILINGUAL_MODEL_PATH]
    
    The purpose of this little script is to compute the needed datasets for the
    training, the validation and the test of the neural networks.
    
    optional arguments:
      -h, --help            show this help message and exit
      --padding             use the "Repeat words and add padding" to fulfill the
                            matrix
      --input INPUT_PATH    cuboid analogies textfile to proceed.
      --output OUTPUT_PATH  output name to store matrices.
      --first_language_model FIRST_MODEL_PATH
                            word embeding model path. Will be open with
                            gensim.open()
      --second_language_model SECOND_MODEL_PATH
                            word embeding model path. Will be open with
                            gensim.open()
      --bilingual_model BILINGUAL_MODEL_PATH
                            word translation model path. Should looks like a
                            trained Hieralign model

 ### `neuralNetwork.py`
This script contains the neural networks architectures and all the needed functions to train and test the neural networks.
The script contains docstring documentation.

## Authors

* **Valentin Taillandier** - [http://valentin-taillandier.fr/](http://valentin-taillandier.fr/)

## License

This project is licensed under the GPL License - see the [https://www.gnu.org/licenses/gpl-3.0.en.html](https://www.gnu.org/licenses/gpl-3.0.en.html) for details.



