import pandas as pd
import torch.optim as optim


from preprocessing import *
from models import *
from train_test_valid import *

columns = ['ID', 'label', 'title', 'Subject', 'Speaker', 'Job Title', 'State', 'Party', 'Barely True', 'False', 'Half True', 'Mostly True', 'Pants on Fire', 'Context']
# LIAR Dataset
train_data = pd.read_csv("Liardataset/train.tsv", sep='\t', names=columns)
test_data = pd.read_csv("Liardataset/test.tsv", sep='\t', names=columns)
valid_data = pd.read_csv("Liardataset/valid.tsv", sep='\t', names=columns)


label_mapping = {'pants-fire': 0, 'false': 0, 'barely-true': 0, 'half-true': 0, 'mostly-true': 1, 'true': 1}
train_data['label'] = train_data['label'].map(label_mapping)
test_data['label'] = test_data['label'].map(label_mapping)
valid_data['label'] = valid_data['label'].map(label_mapping)



#hyper parameters


batch_size = 32
embedding_dim = 155
max_length = 50


data = pd.concat((train_data,test_data),axis=0)
combined_data = pd.concat((data,valid_data), axis=0, ignore_index=True, sort=False)
data = combined_data.sample(frac=1)


data = drop_NA_values(data)


columns_to_drop = ['ID', 'Subject', 'Speaker', 'Job Title', 'State', 'Party', 'Barely True', 'False', 'Half True', 'Mostly True', 'Pants on Fire', 'Context']
data.drop(columns=columns_to_drop, inplace=True)


data['title'] = data['title'].apply(preprocess)

temp_data = data[:int(round((0.8)*len(data)))]
train_data = temp_data[:int(round((0.75)*len(temp_data)))]
valid_data = temp_data[int(round((0.75)*len(temp_data))):]
test_data = data[int(round((0.8)*len(data))):]


# get word arrays
tokenized_list_titles_train = [tokenizer(title) for title in train_data['title']]
labels_train = train_data['label']

tokenized_list_titles_valid = [tokenizer(title) for title in valid_data['title']]
labels_valid = valid_data['label']

tokenized_list_titles_test = [tokenizer(title) for title in test_data['title']]
labels_test = test_data['label']



vocab, frequency_map = generate_frequent_indexed_vocabulary(tokenized_list_titles_train)





print(data["title"][12])



print("-----------Model 1------------")

one_hot_texts_train = [text_to_one_hot(title, vocab) for title in train_data['title']]
one_hot_texts_valid = [text_to_one_hot(title, vocab) for title in valid_data['title']]
one_hot_texts_test = [text_to_one_hot(title, vocab) for title in test_data['title']]



train_dataloader = one_hot_to_tensors(one_hot_texts_train,labels_train,vocab,batch_size)
valid_dataloader = one_hot_to_tensors(one_hot_texts_valid,labels_valid,vocab,batch_size)
test_dataloader = one_hot_to_tensors(one_hot_texts_test,labels_test,vocab,batch_size)

# hyper parameters
input_size = len(vocab)
hidden_size = 64
output_size = 1
model = TextClassifier(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 3
num_layers = 2



training_testing_and_validation(model,criterion,optimizer,num_epochs,train_dataloader,test_dataloader,valid_dataloader)


print("-----------Model 2------------")

train_dataloader = tokenized_titles_to_tensors(tokenized_list_titles_train,labels_train,vocab,batch_size)
valid_dataloader = tokenized_titles_to_tensors(tokenized_list_titles_valid,labels_valid,vocab,batch_size)
test_dataloader = tokenized_titles_to_tensors(tokenized_list_titles_test,labels_test,vocab,batch_size)

model = CustomModel(vocab_size, embedding_dim, max_length)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

training_testing_and_validation(model,criterion,optimizer,num_epochs,train_dataloader,test_dataloader,valid_dataloader)





print("-----------Model 3------------")

train_dataloader = tokenized_titles_to_tensors(tokenized_list_titles_train,labels_train,vocab,batch_size)
valid_dataloader = tokenized_titles_to_tensors(tokenized_list_titles_valid,labels_valid,vocab,batch_size)
test_dataloader = tokenized_titles_to_tensors(tokenized_list_titles_test,labels_test,vocab,batch_size)


hidden_size = 64
num_layers = 2

model = Model_3(vocab_size, embedding_dim, hidden_size, num_layers)


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5



training_testing_and_validation(model,criterion,optimizer,num_epochs,train_dataloader,test_dataloader,valid_dataloader)



print("-----------Model 4------------")

train_dataloader = tokenized_titles_to_tensors(tokenized_list_titles_train,labels_train,vocab,batch_size)
valid_dataloader = tokenized_titles_to_tensors(tokenized_list_titles_valid,labels_valid,vocab,batch_size)
test_dataloader = tokenized_titles_to_tensors(tokenized_list_titles_test,labels_test,vocab,batch_size)


hidden_size = 64
num_layers = 2

model = Model_4(vocab_size, embedding_dim, hidden_size)


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5



training_testing_and_validation(model,criterion,optimizer,num_epochs,train_dataloader,test_dataloader,valid_dataloader)