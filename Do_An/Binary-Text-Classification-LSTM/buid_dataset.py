import os
import nltk
from random import shuffle
import collections


def run(data_folder, data_label_file, num_words = 0, shuffle_data = False):
    # Build the vocabulary and assign a unique number to each word
    vocabulary_dict = word_encoding(data_folder)
    # Encode paragraphs based on the built vocabulary
    # Assign the label of the documents to their corresponding paragraphs
    x_paragraphs, y_paragraphs, y_documents, document_num_paragraph, average_sequence_length = paragraph_encoding(data_folder, vocabulary_dict, data_label_file, num_words, shuffle_data)
    return x_paragraphs, y_paragraphs, y_documents, document_num_paragraph, len(vocabulary_dict) + 1 , average_sequence_length


def paragraph_encoding(data_folder,vocabulary_dict, data_label_file, num_words, shuffle_data):
    label_dict = {}
    with open(data_label_file, "r") as label_file:
        lines = label_file.read().splitlines()
    for line in lines:
        key, value = line.split(" ")
        label_dict[key] = str(value).replace("\r", "").replace("\r\n", "").replace("\n", "")
    labels = [item for item, count in collections.Counter(list(label_dict.values())).items() if count >= 1]
    file_names = os.listdir(data_folder)
    X = []
    y = []
    y_document = []
    document_num_paragraph = []
    if num_words == 0:
        # calculate average paragraph length
        line_count = 0
        word_sum = 0
        for file_name in file_names:
            with open(data_folder + file_name, "r") as f:
                lines = f.readlines()
                for line in lines:
                    word_sum += len(nltk.tokenize.word_tokenize(line))
                    line_count += 1
        average_sequence_length = word_sum / line_count
    else:
        average_sequence_length = num_words
    if shuffle_data:
        shuffle(file_names)
    for file_name in file_names:
        with open(data_folder + file_name, "r") as f:
            lines = f.readlines()
        number_of_paragraphs = 0
        for index, line in enumerate(lines):
            words = nltk.tokenize.word_tokenize(line)
            paragraph_vector = []
            for word in words:
                try:
                    paragraph_vector.append(vocabulary_dict[word])
                except KeyError:
                    paragraph_vector.append(0)
            # Trim paragraph vector if it is larger than average length
            if len(paragraph_vector) > average_sequence_length:
                paragraph_vector = paragraph_vector[:int(average_sequence_length)]
            # Extend paragraph vector by adding zero if it is smaller than average length
            else:
                zero_list = [0] * (int(average_sequence_length) - len(paragraph_vector))
                paragraph_vector += zero_list
            X.append(paragraph_vector)
            # Build labels vector by assigning the label of a document to its corresponding paragraphs
            if label_dict[file_name.split("_")[0]] == labels[1]:
                y.append(1)
            else:
                y.append(0)
            number_of_paragraphs = index + 1
        document_num_paragraph.append(number_of_paragraphs)
        if label_dict[file_name.split("_")[0]] == labels[1]:
            y_document.append(1)
        else:
            y_document.append(0)

    return X, y, y_document, document_num_paragraph, average_sequence_length


def word_encoding(data_folder):
    text = ""
    files = os.listdir(data_folder)
    for file_name in files:
        with open(data_folder + file_name, "r") as f:
            content = f.read()
            text += content
    words = nltk.tokenize.word_tokenize(text)
    vocabulary = collections.Counter(words).most_common()
    vocabulary_dictionary = dict()
    for word, _ in vocabulary:
        # Assign a numerical unique value to each word inside vocabulary
        vocabulary_dictionary[word] = len(vocabulary_dictionary) + 1
    return vocabulary_dictionary
