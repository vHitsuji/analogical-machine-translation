python3 sentences2matrices.py \
--input ../corpora/en-fr.cuboid.nlg \
--output en-fr.matrices \
--first_language_model ../model/wordvectors/FastText/en.kv \
--second_language_model ../model/wordvectors/FastText/fr.kv \
--bilingual_model ../model/bilingual/en_fr_symetric.model 

