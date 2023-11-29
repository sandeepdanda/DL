import torch.nn as nn
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length):
        super(CustomModel, self).__init__()
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        # Fully Connected Layer with 'sigmoid' activation
        self.dense = nn.Sequential(nn.Linear(embedding_dim, 1),
                                   nn.Sigmoid())  # Adding 'sigmoid' activation directly to the Dense layer

    def forward(self, x):
        # Embedding step
        x = self.embedding(x)
        embedded_avg = torch.mean(x, dim=1)
        # Fully connected layer step with 'sigmoid' activation
        output = self.dense(embedded_avg)

        return output
        

class Model_3(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(Model_3, self).__init__()
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
       

    def forward(self, x):
        # Embedding step
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        output, _ = self.lstm(x, (h0, c0))
        output = self.fc(output[:, -1, :])
        output = self.sigmoid(output)

        return output
        
        
class Model_4(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Model_4, self).__init__()
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
       

    def forward(self, x):
        # Embedding step
        x = self.embedding(x)
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        output = self.sigmoid(output)

        return output
        
        
class TextClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x