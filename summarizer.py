import nltk
import math
from nltk.corpus import stopwords
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import operator

#Remove stopwords (like "the" or "is") from a sentence

def removeStopwords(sentence):
    stops = stopwords.words('english')
    newSentence = " ".join(word for word in sentence if word not in stops)
    return newSentence

#TF-IDF

def tf(word, sen):
    if len(sen.split()) == 0:
        length = 0.01
    else:
        length = len(sen.split())
    return sen.split().count(word)/length

def idf(word, cleanSentences):
    appears = 0
    for sentence in cleanSentences:
        if word in sentence.split():
            appears += 1
            continue
    return math.log10(len(cleanSentences)/appears)

#Function to determine the cosine similarity between two sentence vectors

def cosineSimilarity(sen1, sen2):
    product = 0
    for i in range(len(sen1)):
        product += sen1[i] * sen2[i]
    return product/(np.linalg.norm(sen1) * np.linalg.norm(sen2))

#Main function

def summarize(text):

    #The number of sentences to be printed out for the summary

    TOP_RANKED = 7

    # Separate the text into sentences and return them in an array of sentences

    sentences = nltk.tokenize.sent_tokenize(text)

    # Clean the sentences: make everything lowercase, eliminate any symbol, except the ' symbol and remove the stopwords

    cleanSentences = [sentence.lower() for sentence in sentences]

    cleanSentences = [''.join(letter for letter in sentence if
                              not letter.isdigit() and letter not in [',', '.', '"', ':', ';', '!', '?', '”', '“', '-',
                                                                      '&']) for sentence in cleanSentences]

    cleanSentences = [removeStopwords(sentence.split()) for sentence in cleanSentences]

    # Make a set of all the words found in the cleanSentences

    words = set()

    for sentence in cleanSentences:
        for word in sentence.split():
            words.add(word)

    # Construct the sentenceVectors, i.e. calculate the TF-IDF for each word in relation to each sentence
    # The final array will be of size equal to the number of sentences

    sentenceVectors = []

    for sentence in cleanSentences:
        sentenceVector = []
        for word in words:
            sentenceVector.append(tf(word, sentence) * idf(word, cleanSentences))
        sentenceVectors.append(sentenceVector)

    # Construct the similarityMatrix between the sentences using the cosineSimilarity function

    similarityMatrix = np.zeros([len(cleanSentences), len(cleanSentences)])

    for i in range(len(cleanSentences)):
        for j in range(len(cleanSentences)):
            if i != j:
                similarityMatrix[i][j] = cosineSimilarity(sentenceVectors[i], sentenceVectors[j])

    # graph = nx.from_numpy_array(similarityMatrix)
    # nx.draw_networkx(graph)
    # plt.show()

    # Construct the finalScores
    # For each sentence, we calculate the sum of the incoming edges score

    finalScores = {}
    for j in range(len(cleanSentences)):
        score = 0
        for i in range(len(cleanSentences)):
            if i != j:
                score += similarityMatrix[i][j]
        finalScores[j] = score

    # After we calculate the finalScores, we normalize the final vector so that the sum of all the sentence scores equals to 1

    finalScoresNormalize = sum(list(finalScores.values()))
    for key in finalScores.keys():
        finalScores[key] = finalScores[key] / finalScoresNormalize

    # We construct the final summary using the first TOP_RANKED sentences

    summary = ""
    ranked = sorted(finalScores.items(), key=operator.itemgetter(1), reverse=True)

    for i in range(TOP_RANKED):
        summary += sentences[ranked[TOP_RANKED - i - 1][0]] + '\n'

    return summary

# print(words)
# print(words)
# print(len(cleanSentences))
# print(cleanSentences)
# print(sentenceVectors)
# print(similarityMatrix)
# print(sorted(finalScores.items(), key=operator.itemgetter(1), reverse=True))

# file = open("text.txt")
# text = file.read()
# file.close()
#
# summarize(text)
