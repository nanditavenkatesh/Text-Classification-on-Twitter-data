# Text Classification on Twitter Data

## Overview
This project implements and compares two text classification algorithms, **Naive Bayes** and **Logistic Regression**, on a dataset of social media posts from Twitter. The goal is to predict the sentiment (positive, neutral, or negative) of tweets using natural language processing (NLP) and machine learning techniques.

The project progresses through various stages, from basic preprocessing of raw text data to implementing more elaborate preprocessing steps. Finally, we analyze the performance of the classifiers and propose improvements to the workflow.

## Project Structure
- **Data**: The dataset used consists of tweets from a CSV file (`Tweets_5K.csv`). Each tweet is labeled with a sentiment: positive (1), neutral (0), or negative (-1).
- **Main Algorithms**:
  - Naive Bayes Classifier
  - Logistic Regression Classifier
- **Preprocessing Methods**:
  - Basic Tokenization
  - Advanced Preprocessing (including lowercasing, lemmatization, removing stop words, etc.)

## Tasks
The project is broken down into the following main tasks:

### Part 1: Data Preparation
- **Raw Data Loading**: Tweets are loaded from the dataset along with their corresponding sentiment labels.
- **Basic Preprocessing**: Tokenization of tweets by splitting on whitespace.
- **Feature Engineering**: Creation of a Bag of Words (BoW) model with Laplace Add-1 smoothing.
- **Train-Test Split**: Data is split into 80% training and 20% testing sets.

### Part 2: Naive Bayes Classifier
- **Implementation**: Naive Bayes classification using `scikit-learn`.
- **Evaluation**: Model accuracy is reported on the test set.
- **Manual Classification Example**: A Naive Bayes classification is manually computed for a sample tweet ("Happy birthday Annemarie").

### Part 3: Logistic Regression Classifier
- **Implementation**: Logistic Regression is implemented using `scikit-learn` with a one-vs-rest scheme for multiclass classification.
- **Evaluation**: Model accuracy is compared against the Naive Bayes classifier and a baseline classifier that predicts the most frequent category.

### Part 4: Advanced Preprocessing
- **Preprocessing Steps**: Additional preprocessing steps such as lowercasing, lemmatization, removing punctuation, handling stop words, and reducing the feature space to the 1000 most frequent words (with an OOV token for others).
- **Evaluation**: The impact of advanced preprocessing on the Naive Bayes and Logistic Regression classifiers is analyzed.

## Requirements
- Python 3.x
- Libraries:
  - `scikit-learn`
  - `spaCy`
  - `nltk`
  
## How to Run
1. Install the required packages.
2. Preprocess the data by following the steps outlined in the code.
3. Train the models (Naive Bayes and Logistic Regression) using the preprocessed data.
4. Evaluate the performance of the models on the test set.

## Results
- **Naive Bayes Accuracy**: 0.434 achieved on the test set.
- **Logistic Regression Accuracy**: 0.633 achieved on the test set.
  
## Additional Improvements
- Implemented an additional feature and evaluated its effect on performance.
- Analyzed incorrectly classified tweets to identify patterns and suggest further improvements.

## References
1. Pandas. (n.d.). pandas.read_csv — Pandas 1.2.4 Documentation. Pandas.pydata.org. https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

2. Here is how to select multiple columns in a Pandas dataframe in Python. (n.d.). Pythonhow.com. Retrieved February 2, 2024, from https://pythonhow.com/how/select-multiple-columns-in-a-pandas-dataframe/#:~:text=In%20summary%2C%20to%20select%20multiple

3. sklearn.feature_extraction.DictVectorizer. (n.d.). Scikit-Learn. Retrieved February 2, 2024, from https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer.get_feature_names_out

4. PhD, E. G. (2023, November 11). Understanding Multinomial Naive Bayes Classifier. Medium. https://medium.com/@evertongomede/understanding-multinomial-naive-bayes-classifier-fdbd41b405bf

5. python - Find the item with maximum occurrences in a list. (n.d.). Stack Overflow. https://stackoverflow.com/questions/6987285/find-the-item-with-maximum-occurrences-in-a-list

6. python - How do I count the occurrences of a list item? (n.d.). Stack Overflow. https://stackoverflow.com/questions/2600191/how-do-i-count-the-occurrences-of-a-list-item

7. Logistic regression in Python with Scikit-learn. (2022, September 15). Mlnuggets. https://www.machinelearningnuggets.com/logistic-regression/

8. Tracyrenee. (2021, September 18). Conduct a Twitter Sentiment Analysis Test using Spacy. MLearning.ai. https://medium.com/mlearning-ai/conduct-a-twitter-sentiment-analysis-test-using-spacy-7715db4b2c97

9. Twitter Sentiment Analysis using Spacy. (n.d.). Kaggle.com. Retrieved February 2, 2024, from https://www.kaggle.com/code/mansimeena/twitter-sentiment-analysis-using-spacy

10. Chang, E. (2021, November 28). Complete Guide to Perform Classification of Tweets with SpaCy. Medium. https://towardsdatascience.com/complete-guide-to-perform-classification-of-tweets-with-spacy-e550ee92ca79

11. How do I get indices of N maximum values in a NumPy array? (n.d.). Stack Overflow. Retrieved February 2, 2024, from https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array

12. Python | Remove punctuation from string. (2020, March 30). GeeksforGeeks. https://www.geeksforgeeks.org/python-remove-punctuation-from-string/#

