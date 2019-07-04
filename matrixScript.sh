python3 sentences2matrices.py \
--input ../../corpora/train+tune.en-fr.nlg \
--output en-fr.matrices \
--first_language_model ./wordvectors/FastText/en.kv \
--second_language_model ./wordvectors/FastText/fr.kv \
--bilingual_model bilingual/stored_model


