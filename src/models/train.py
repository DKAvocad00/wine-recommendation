if __name__ == '__main__':
    import argparse
    from gensim.models.doc2vec import Doc2Vec
    from src.utils.file_utils import read_csv
    from src.constants import *
    from src.utils.preprocess_utils import preprocess_data
    from src.utils.summarize_utils import summarize_data
    from src.utils.utils import tagged_documents, ProgressCallback

    parser = argparse.ArgumentParser(description="Train a Doc2Vec model with hyperparameters.")

    parser.add_argument("--vector_size",
                        default=VECTOR_SIZE,
                        type=int,
                        help="Dimensionality of the feature vectors.")

    parser.add_argument("--window_size",
                        default=WINDOW_SIZE,
                        type=int,
                        help="The maximum distance between the current and predicted word within a sentence.")

    parser.add_argument("--learning_rate",
                        default=LEARNING_RATE,
                        type=float,
                        help="The initial learning rate.")

    parser.add_argument("--min_learning_rate",
                        default=MIN_LEARNING_RATE,
                        type=float,
                        help="Learning rate will linearly drop to min_alpha as training progresses.")

    parser.add_argument("--min_count",
                        default=MIN_COUNT,
                        type=int,
                        help="Ignores all words with total frequency lower than this.")

    parser.add_argument("--num_epochs",
                        default=NUM_EPOCHS,
                        type=int,
                        help="The number of epochs to train for")

    args = parser.parse_args()

    VECTOR_SIZE = args.vector_size
    WINDOW_SIZE = args.window_size
    LEARNING_RATE = args.learning_rate
    MIN_LEARNING_RATE = args.min_learning_rate
    MIN_COUNT = args.min_count
    NUM_EPOCHS = args.num_epochs

    print(f"[INFO] Training a Doc2Vec model with hyperparameters:\n"
          f" vector_size: {VECTOR_SIZE}\n"
          f" window_size: {WINDOW_SIZE}\n"
          f" learning_rate: {LEARNING_RATE}\n"
          f" min_learning_rate: {MIN_LEARNING_RATE}\n"
          f" min_count: {MIN_COUNT}\n"
          f" num_epochs: {NUM_EPOCHS}")

    print("[INFO] Reading data file...")
    reviews = read_csv(DATA_FILE_PATH, drop_duplicates=True)

    print("[INFO] Cleaning data...")
    reviews = preprocess_data(reviews, 'description', 'description_cleaned')

    print("[INFO] Summarizing data...")
    reviews = summarize_data(reviews, 'description_cleaned', 'summary')

    print("[INFO] Training Doc2Vec model...")
    documents = tagged_documents(reviews['summary'])

    model = Doc2Vec(vector_size=VECTOR_SIZE,
                    window=WINDOW_SIZE,
                    min_count=MIN_COUNT,
                    workers=8,
                    alpha=LEARNING_RATE,
                    min_alpha=MIN_LEARNING_RATE,
                    dm=0)

    model.build_vocab(documents)
    model.train(documents,
                total_examples=len(documents),
                epochs=NUM_EPOCHS,
                callbacks=[ProgressCallback(total_epochs=NUM_EPOCHS)])

    print("[INFO] Saving data...")
    reviews.to_csv('../../data/wine_reviews_final.csv', index=False)

    print("[INFO] Saving model...")
    model.save('../../model/doc2vec_model')

    print("[INFO] Model saved")
