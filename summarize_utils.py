from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest
from nltk import sent_tokenize


def split_data(texts):
    if isinstance(texts, str):
        sentences = sent_tokenize(texts)
    else:
        sentences = [sentence for text in texts for sentence in sent_tokenize(text)]

    return sentences


def generate_summary(texts):
    sentences = split_data(texts)
    if len(sentences) < 2:
        return sentences

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    summary_sentences = nlargest(7, range(len(sentence_scores)), key=sentence_scores.__getitem__)

    summary_tfidf = ' '.join([sentences[i] for i in sorted(summary_sentences)])

    return summary_tfidf


def summarize_data(data, column, summarize_column):
    data[summarize_column] = data[column].apply(lambda x: generate_summary(x))

    return data