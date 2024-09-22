if [ ! -f glove.840B.300d.txt ]; then
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove.840B.300d.zip
  unzip glove.840B.300d.zip
fi
rm  -f glove.840B.300d.zip

python intent_preprocess.py
python slot_preprocess.py

