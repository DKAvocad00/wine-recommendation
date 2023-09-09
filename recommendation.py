if __name__ == '__main__':
    import argparse
    from sklearn.metrics.pairwise import cosine_similarity
    from gensim.models.doc2vec import *
    from file_utils import *
    from constants import *
    from summarize_utils import generate_summary
    from preprocess_utils import preprocess_text
    from utils import *

    parser = argparse.ArgumentParser(description="Outputs the top n similar wines on description")

    parser.add_argument("--description",
                        default="Much like the regular bottling from 2012, this comes across as rather rough and "
                                "tannic, with rustic, earthy, herbal characteristics. Nonetheless, if you think of it "
                                "as a pleasantly unfussy country wine, it's a good companion to a hearty winter stew.",
                        type=str,
                        help="Description to search similar wines for")

    args = parser.parse_args()

    DESCRIPTION = args.description

    print(f" [INFO] Recommendation similar wines with parameters:\n"
          f" description: {DESCRIPTION}")

    print(f" [INFO] Preprocessing description...")
    description = preprocess_text(DESCRIPTION)

    print(f" [INFO] Summarizing description...")
    summary = generate_summary(description)

    print("Summarized description: {}".format(summary))

    print(f" [INFO] Loading doc2vec model...")
    model = Doc2Vec.load('model/doc2vec_model')

    print(f" [INFO] Loading doc2vec model...")
    data = read_file(FINAL_DATA_FILE_PATH)
    print(f" [INFO] Getting similar wines...")
    similar_wines = get_similar_wines(summary, model, data)

    columns_to_display = ['points', 'title', 'description', 'price']

    print(f"\nSummary to Search:\n{summary}\n")
    print(f"\nTop 5 Similar Descriptions:\n")
    for idx, row in similar_wines.iterrows():
        print(f"doc {idx + 1}:")
        for col in columns_to_display:
            print(f"{col}: {row[col]}")
        print()
