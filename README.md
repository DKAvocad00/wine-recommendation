# ML Project: Wine Recommendation

This project is designed to help users find wines similar to their preferences by analyzing wine descriptions.

## Getting Started

To get started with this project, follow the instructions below.

### Prerequisites

Make sure you have Python installed. Additionally, you'll need to install the required Python packages:

    pip install -r requirements.txt

# Usage
## Getting Summary of a Wine Description

To generate a summary from a wine description, you can run the following command:

    python summarize.py --description "This dry and restrained wine offers spice in profusion. Balanced with acidity and a firm texture, it's very much for food."

Alternatively, you can use the default settings by running:

    python summarize.py

## Finding Top 5 Similar Wines

To discover the top 5 wines similar to your input description, use the following command:

    python recommendation.py --description "This has great depth of flavor with its fresh apple and pear fruits and touch of spice. It's off dry while balanced with acidity and a crisp texture. Drink now."

By default, the script will find similar wines using default settings:

    python recommendation.py 


## Option

--description: Specifies the wine description to be analyzed.




