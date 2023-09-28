import os

# file constants
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE_PATH = os.path.join(BASE_PATH, 'data', 'wine_reviews.csv')
FINAL_DATA_FILE_PATH = os.path.join(BASE_PATH, 'data', 'wine_reviews_final.csv')
DOC2VEC_MODEL_FILE_PATH = os.path.join(BASE_PATH, 'model', 'doc2vec_model')

# model constants
VECTOR_SIZE = 100
WINDOW_SIZE = 5
MIN_COUNT = 1
NUM_EPOCHS = 1000
LEARNING_RATE = 0.025
MIN_LEARNING_RATE = 0.01
