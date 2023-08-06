import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk import pos_tag


# Load and preprocess data
def preprocess(file_path):
    with open(file_path, 'r') as file:
        text = file.read().lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words('english') and token not in string.punctuation and token.isalpha()]
    return tokens

# Function to get most common words
def get_most_common(tokens, n):
    freq_dist = nltk.FreqDist(tokens)
    return freq_dist.most_common(n)

# Function to get pos tags
def get_pos_tags(tokens):
    return pos_tag(tokens)

# Function to get common elements
def common_elements(list1, list2):
    return [element for element in list1 if element in list2]

def show_context(word, tokens, window=6):
    text = nltk.Text(tokens)
    if word in text:
        occurrences = list(text.concordance_list(word, lines=window))
        for occurrence in occurrences:
            print(' '.join(occurrence[0]), ' --> ', word, ' <-- ', ' '.join(occurrence[2]))
    else:
        print(f"The word {word} does not exist in the dataset")

# Load and preprocess datasets
tokens_q = preprocess('Q dataset.txt')
tokens_k1 = preprocess('K1 dataset.txt')
tokens_k2 = preprocess('K2 dataset.txt')

# Get the 20 most common words from each dataset
most_common_q = get_most_common(tokens_q, 20)
most_common_k1 = get_most_common(tokens_k1, 20)
most_common_k2 = get_most_common(tokens_k2, 20)

# Get the common words in the top 20 between Q and K1, and Q and K2
common_q_k1 = common_elements([word for word, count in most_common_q], [word for word, count in most_common_k1])
common_q_k2 = common_elements([word for word, count in most_common_q], [word for word, count in most_common_k2])

# Display the shared words in the top 20
print(f"Top 20 words in Q dataset: {most_common_q}")
print(f"Top 20 words in K1 dataset: {most_common_k1}")
print(f"Top 20 words in K2 dataset: {most_common_k2}")

# Display the shared words in the top 20 between Q and K1, and Q and K2
print(f"Shared words in the top 20 between Q and K1: {common_q_k1}")
print(f"Shared words in the top 20 between Q and K2: {common_q_k2}")

# Compare the number of common words
if len(common_q_k1) > len(common_q_k2):
    print("K1 is more similar to Q based on the top 20 words")
elif len(common_q_k1) < len(common_q_k2):
    print("K2 is more similar to Q based on the top 20 words")
else:
    print("K1 and K2 are equally similar to Q based on the top 20 words")

# Get pos tags for the datasets
pos_tags_q = get_pos_tags(tokens_q)
pos_tags_k1 = get_pos_tags(tokens_k1)
pos_tags_k2 = get_pos_tags(tokens_k2)

# Get common pos tags in the top 20 between Q and K1, and Q and K2
common_pos_q_k1 = common_elements(pos_tags_q, pos_tags_k1)
common_pos_q_k2 = common_elements(pos_tags_q, pos_tags_k2)


# Compare the number of common pos tags
if len(common_pos_q_k1) > len(common_pos_q_k2):
    print("In terms of POS tags, K1 is more similar to Q")
elif len(common_pos_q_k1) < len(common_pos_q_k2):
    print("In terms of POS tags, K2 is more similar to Q")
else:
    print("In terms of POS tags, K1 and K2 are equally similar to Q")

word = input("Plese input the word:")
# Show the context of the word "figure" in Q dataset
show_context(word, tokens_q)

# Show the context of the word "figure" in K1 dataset
show_context(word, tokens_k1)

# Show the context of the word "figure" in K2 dataset
show_context(word, tokens_k2)
