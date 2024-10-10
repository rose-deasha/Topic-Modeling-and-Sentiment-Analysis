# Survey Response Analysis using Topic Modeling and Sentiment Analysis

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Usage](#usage)
6. [Features](#features)
7. [Output](#output)
8. [Customization](#customization)
9. [Results](#results)
10. [Improvements](#improvements)
11. [Troubleshooting](#troubleshooting)
12. [Contributing](#contributing)
13. [License](#license)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rose-deasha/Question.git/HEAD)

---

## Overview

This project focuses on analyzing unstructured survey response data using natural language processing (NLP) techniques. The Jupyter notebook processes text data to extract meaningful insights, identify prevalent topics, and assess sentiment. It employs Python libraries such as sklearn, gensim, nltk, and visualization tools like matplotlib and seaborn to help visualize and quantify the survey insights.

---

## Prerequisites

To run this project, you need:
- Python 3.7+
- Jupyter Notebook

---

## Installation

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

Note: The `requirements.txt` file should include:
```
numpy
matplotlib
seaborn
scikit-learn
wordcloud
nltk
gensim
jupyter
```

---

## Project Structure

The project consists of a single Jupyter notebook (`Question.ipynb`) containing all the analysis code. Here's a breakdown of its main components:

1. Data Loading and Preprocessing
2. Topic Modeling using LDA
3. Topic Visualization
4. Sentiment Analysis using VADER
5. Word Cloud Generation
6. Topic Model Validation using Coherence Score

---

## Usage

1. Open the `Question.ipynb` notebook in Jupyter.
2. Run all cells in the notebook to perform the analysis.

---

## Features

### 1. Topic Modeling
- Uses Latent Dirichlet Allocation (LDA) to identify main topics in the survey responses.
- Visualizes topic distribution across all responses.
- Creates a heatmap showing topic distribution for individual responses.

### 2. Sentiment Analysis
- Employs VADER sentiment analyzer to assess the sentiment of each response.
- Generates a histogram of sentiment scores.

### 3. Word Cloud Generation
- Creates word clouds for each identified topic, providing a visual representation of key terms.

### 4. Model Validation
- Calculates the coherence score of the LDA model to evaluate its effectiveness.

---

## Output

The script generates several output files:

1. `topic_distribution.png`: Bar chart of average topic distribution.
2. `topic_heatmap.png`: Heatmap of topic distribution across responses.
3. `vader_sentiment_distribution.png`: Histogram of sentiment scores.
4. `topic_X_wordcloud.png`: Word cloud for each topic (where X is the topic number).

---

## Customization

You can customize the analysis by modifying the following parameters:

- `num_topics`: Number of topics for LDA (default is 5)
- `num_top_words`: Number of top words to display for each topic
- Adjust preprocessing steps in the `preprocess_text` function

---

## Results

The analysis provides insights into key areas of customer feedback, including:
- Prevalent topics in survey responses
- Sentiment distribution across responses
- Visualizations of key terms and topics

---

## Improvements

Potential improvements to the project:
- Fine-tuning of LDA parameters for better topic coherence
- Implementing more advanced text preprocessing techniques
- Adding interactive visualizations for dynamic data exploration

---

## Troubleshooting

If you encounter issues:
1. Ensure all required packages are installed correctly.
2. Check that you're using a compatible Python version.
3. Verify that the input data is in the correct format.

---

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes.
4. Push to the branch.
5. Create a new Pull Request.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
