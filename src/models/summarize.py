if __name__ == '__main__':
    import argparse
    from src.utils.preprocess_utils import preprocess_text
    from src.utils.summarize_utils import generate_summary

    parser = argparse.ArgumentParser(description="Outputs the summary of the input description")

    parser.add_argument("--description",
                        default="A snappy, high-acid wine that pulses with red fruit aromas and flavors."
                                "It's sheering, which is normal for the Graciano variety,"
                                "but it also tastes nice while delivering a bolt of raspberry and red plum flavor. "
                                "Oak comes up late, providing a counterweight of vanilla and dry spice. "
                                "Nice but like all varietal Gracianos, it's sharp and limited.",
                        type=str,
                        help="The description to summary for")

    args = parser.parse_args()

    DESCRIPTION = args.description

    print(f" [INFO] Summarize description with parameters:\n"
          f" description: {DESCRIPTION}")

    print(f" [INFO] Preprocessing description...")
    description = preprocess_text(DESCRIPTION)

    print(f" [INFO] Summarizing description...")
    summary = generate_summary(description)

    print("Summarized description: {}".format(summary))




