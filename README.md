# Survey Insights Engine: Advanced Topic Modeling and Sentiment Analysis

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Output](#output)
7. [Improvements](#improvements)
8. [Customization](#customization)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)
11. [License](#license)

---

## Overview

The Survey Insights Engine is an advanced tool for analyzing unstructured survey response data using state-of-the-art natural language processing (NLP) techniques. This updated version offers improved topic modeling, enhanced preprocessing, and more comprehensive sentiment analysis to extract meaningful insights from text data.

---

## Key Features

1. **Improved Text Preprocessing**
   - Enhanced tokenization and lemmatization
   - Customizable stop words removal, including domain-specific terms
   - Retention of meaningful short words for better context preservation

2. **Advanced Topic Modeling**
   - Latent Dirichlet Allocation (LDA) with optimized hyperparameters
   - Non-Negative Matrix Factorization (NMF) for alternative topic extraction
   - Improved coherence scores for more meaningful topic identification

3. **Comprehensive Sentiment Analysis**
   - VADER sentiment analyzer for nuanced sentiment scoring
   - Visualization of sentiment distribution across responses

4. **Enhanced Visualizations**
   - Topic distribution charts for both LDA and NMF models
   - Interactive heatmaps for topic distribution across responses
   - Word clouds for easy identification of key terms in each topic

5. **Model Validation and Comparison**
   - Coherence score calculation for both LDA and NMF models
   - Comparative analysis between LDA and NMF results

6. **Flexible Data Handling**
   - Support for various input formats, including CSV files and lists

---

## Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab

---

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/survey-insights-engine.git
   ```
2. Navigate to the project directory:
   ```
   cd survey-insights-engine
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

---

## Usage

1. Open `survey_insights_engine.ipynb` in Jupyter Notebook or JupyterLab.
2. Update the `survey_responses` variable with your data or specify the path to your CSV file.
3. Run all cells in the notebook to perform the analysis.

---

## Output

The engine generates several output files:

1. `lda_topic_distribution.png` & `nmf_topic_distribution.png`: Bar charts of average topic distribution for LDA and NMF models.
2. `lda_topic_heatmap.png` & `nmf_topic_heatmap.png`: Heatmaps of topic distribution across responses.
3. `vader_sentiment_distribution.png`: Histogram of sentiment scores.
4. `topic_X_wordcloud.png`: Word clouds for each topic or an overall word cloud (where X is the topic number).

---

## Improvements

This version introduces several key improvements:

1. **Enhanced Coherence**: By implementing both LDA and NMF models, we achieve better topic coherence and interpretability. The coherence scores are now calculated for both models, allowing for direct comparison and selection of the most appropriate model for your data.

2. **Improved Preprocessing**: The new preprocessing pipeline includes lemmatization and more nuanced stop word removal, resulting in more meaningful topic extraction.

3. **Dual Topic Modeling**: The addition of NMF alongside LDA provides a complementary perspective on topic extraction, often resulting in more interpretable topics for certain types of data.

4. **Extended Visualizations**: New visualizations for NMF results and comparative views between LDA and NMF offer deeper insights into the topic structure of your data.

5. **Flexible Hyperparameter Tuning**: Easily adjustable hyperparameters for both LDA and NMF models allow for fine-tuning based on your specific dataset.

---

## Customization

You can customize the analysis by modifying:

- `num_topics`: Number of topics for LDA and NMF (default is 5)
- `doc_topic_prior` and `topic_word_prior`: Hyperparameters for LDA
- `max_df` and `min_df` in TfidfVectorizer for vocabulary building
- `ngram_range` to include multi-word phrases in the analysis

---

## Troubleshooting

If you encounter issues:
1. Ensure all required packages are installed correctly.
2. Verify that your input data is in the correct format.
3. Check the Jupyter notebook for any error messages and refer to the documentation of the relevant libraries.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
