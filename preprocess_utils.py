import spacy
import re
import nltk.corpus


def _normalize_text(text):
    tm1 = re.sub('<pre>.*?</pre>', '', text, flags=re.DOTALL)
    tm2 = re.sub('<code>.*?</code>', '', tm1, flags=re.DOTALL)
    tm3 = re.sub('<[^>]+>©', '', tm2, flags=re.DOTALL)
    return tm3.replace("\n", "")


def _cleanup_text(text, nlp, stopwords):
    punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~©'
    doc = nlp(text, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
    tokens = ' '.join(tokens)
    return tokens


def preprocess_text(text, nlp=spacy.load('en_core_web_lg'), stopwords=nltk.corpus.stopwords.words('english')):
    text = _normalize_text(text)
    text = _cleanup_text(text, nlp, stopwords)

    return text


def preprocess_data(data, column, preprocess_column):

    data[preprocess_column] = data[column].apply(lambda x: preprocess_text(x))

    return data
