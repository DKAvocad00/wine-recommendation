from gensim.models.doc2vec import TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
from nltk.tokenize import word_tokenize


class ProgressCallback(CallbackAny2Vec):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch = 0

    def on_epoch_begin(self, model):
        print(f"Epoch {self.epoch + 1}/{self.total_epochs} - Training: ", end='', flush=True)

    def on_epoch_end(self, model):
        print("Completed")
        self.epoch += 1


def tagged_documents(data):
    documents = []

    data = data.astype(str).tolist()
    splitted_texts = [text.split() for text in data]
    idx = [i for i in range(len(data))]
    for i in range(len(data)):
        documents.append(TaggedDocument(splitted_texts[i], [idx[i]]))

    return documents


def get_similar_wines(summary, model, data):
    summary = word_tokenize(summary)
    vec = model.infer_vector(summary)
    similar_indices = model.dv.most_similar([vec], topn=5)
    similar_indices = [idx for idx, _ in similar_indices]
    similar_descriptions = data.iloc[similar_indices]

    return similar_descriptions


