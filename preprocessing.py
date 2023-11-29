import pandas as pd
import string # for removing punctuations
import re # for tokenization
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torchtext.data import get_tokenizer



nltk.download('stopwords')
stopwords = stopwords.words('english')


tokenizer = get_tokenizer("basic_english")


nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()


vocab_size = 5000




# dropping NA
def drop_NA_values(data):
  data.dropna(inplace=True)
  return data
  
  
  
def preprocess(text):
  text = "".join([i for i in text if i not in string.punctuation])
  text = text.lower()
  tokens = tokenizer(text)
  filtered_words = [i for i in tokens if i not in stopwords]
  lemmatized_text = [wordnet_lemmatizer.lemmatize(word) for word in filtered_words]
  joined_sentence = " ".join(lemmatized_text)
  cleaned_sentence = re.sub(' +', ' ', joined_sentence)
  return cleaned_sentence
  
  
  
def generate_frequent_indexed_vocabulary(tokenized_list_titles):
  vocab_dictionary = {}

  vocab_dictionary["<PAD>"] = 0
  vocab_dictionary["<UNK>"] = 1
  word_start_index = 2

  frequency_map={}

  for tokenized_title in tokenized_list_titles:
    for token in tokenized_title:
      if token not in frequency_map:
        frequency_map[token] = 1
      else:
        frequency_map[token] += 1

  sorted_frequency_map = sorted(frequency_map.keys(), key=lambda word: frequency_map[word], reverse=True)
  # sorted_frequency_map = sorted(frequency_map.keys(), key=lambda word: frequency_map[word])
  
  for word in sorted_frequency_map[:vocab_size - 2]:
        vocab_dictionary[word] = word_start_index
        word_start_index += 1


  return vocab_dictionary, frequency_map
  
  
  
def tokenized_titles_to_tensors(tokenized_list_titles,labels,vocab,batch_size):
  indexed_titles = [[vocab.get(token, vocab["<UNK>"]) for token in tokenized_title] for tokenized_title in tokenized_list_titles]

  title_tensors = [torch.tensor(indexed_title) for indexed_title in indexed_titles]
  title_tensors_padded = pad_sequence(title_tensors, batch_first=True, padding_value=0)
  print(title_tensors_padded.shape[1])

  title_tensor = torch.tensor(title_tensors_padded, dtype=torch.long)
  label_tensor = torch.tensor(labels.values, dtype=torch.float32)

  dataset = TensorDataset(title_tensor, label_tensor)

  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  return dataloader
  
  
def text_to_one_hot(text, word_to_index):
    one_hot = np.zeros(len(word_to_index))
    for word in text.split():
        if word in word_to_index:
            one_hot[word_to_index[word]] = 1
    return one_hot
    
    
def one_hot_to_tensors(one_hot_texts,labels,vocab,batch_size):
  title_tensor = torch.tensor(one_hot_texts, dtype=torch.float32)
  # label_tensor = torch.tensor(labels.values, dtype=torch.float32).view(-1,1)
  label_tensor = torch.tensor(labels.values, dtype=torch.float32)
  dataset = TensorDataset(title_tensor, label_tensor)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  return dataloader