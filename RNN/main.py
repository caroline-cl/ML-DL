import torch
import random
from utils import *
import time
import pdb
import json
import math


### Original RNN ###
n_letters=58
n_categories=18
n_hidden = 128
n_epochs = 100
print_every = 5000
plot_every = 1000
learning_rate = 0.00002 # If you set this too high, it might explode. If too low, it might not learn

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_shape (int): size of the 1-hot embeddings for each character (this will be 58)
            hidden_layer_width (int): number of nodes in the single hidden layer within the model
            n_classes (int): number of output classes
        """
        super(RNN, self).__init__()
        ### TODO Implement the network architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_layer = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.output_layer = torch.nn.Linear(input_size + hidden_size, output_size)
        # self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, input, hidden):
        """Forward function accepts tensor of input data, returns tensor of output data.
        Modules defined in constructor are used, along with arbitrary operators on tensors
        """
        
        ### TODO Implement the forward function
        comb = torch.cat((input, hidden), 1)
        hidden = self.hidden_layer(comb)
        output = self.output_layer(comb)
        output = torch.nn.functional.softmax(output, dim=1) #+ 1e-8

        #your function will return the output y(t) and hidden h(t) from equation 1 in the docs
        return output, hidden 

    def initHidden(self):
        """
        This function initializes the first hidden state of the RNN as a zero tensor.
        """
        return torch.zeros(1, self.hidden_size)

def get_xy_pairs(names):
    #TODO 
    #process the names dict and convert into a list of (x,y) pairs. x is a 1-hot tensor of size (num_characters_in_name, 1, n_letters)
    #y is a scalar representing the category of the language, there are 18 languages, assign an index between 0-17 to each language and y represents this index.
    #you may make use of the nameToTensor() function in the utils.py file to help you with this function
    list_of_pairs = []
    y = 0
    for key, val in names.items():
        for word in val:
            list_of_pairs.append((nameToTensor(word), torch.tensor(y)))
        y += 1
    return list_of_pairs

def create_train_and_test_set(list_of_pairs):
    #TODO 
    #process the list of (x,y) pairs and split them 80-20 into train and test set
    #train_x is a list of name embeddings each of size (num_characters_in_name, 1, n_letters), train_y is the correponding list of language category index. Same for test_x and test_y
    random.seed(4)
    random.shuffle(list_of_pairs)
    l = len(list_of_pairs)
    train = list_of_pairs[:int(l * 0.8)]
    test = list_of_pairs[int(l * 0.8):]
    train_x, train_y, test_x, test_y = [], [], [], []
    for pair in train:
        train_x.append(pair[0])
        train_y.append(pair[1])
    
    for pair in test:
        test_x.append(pair[0])
        test_y.append(pair[1])
    return train_x, train_y, test_x, test_y

rnn = RNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = torch.nn.NLLLoss()

def train(train_x, train_y):
    """train_x and train_y are lists with names and correspoonding labels"""
    loss = 0
    cnt = 0
    for x, y in zip(train_x, train_y):
        hidden = rnn.initHidden()
        for i in range(x.size()[0]):
            output, hidden = rnn(x[i], hidden)
        loss += criterion(torch.log(output), y.unsqueeze(0)) #the unsqueeze converts the scalar y to a 1D tensor
        cnt += 1
        # pdb.set_trace()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()/cnt

def test(train_x, train_y):
    """train_x and train_y are lists with names and correspoonding labels"""
    loss = 0
    cnt = 0
    y_prediction = []
    with torch.no_grad(): 
        for x, y in zip(train_x, train_y):
            hidden = rnn.initHidden()
            for i in range(x.size()[0]):
                output, hidden = rnn(x[i], hidden)
            y_pred = torch.log(output)
            loss += criterion(y_pred, y.unsqueeze(0)) #the unsqueeze converts the scalar y to a 1D tensor
            y_prediction.append(y_pred)
            cnt += 1
            # pdb.set_trace()
    return loss.item()/cnt, y_prediction


# Keep track of losses for plotting
current_loss = 0
all_losses_train = []
all_losses_test = []
y_predicted = []

#names is your dataset in python dictionary form. Keys are languages and values are list of words belonging to that language
with open('names.json', 'r') as fp:
    names = json.load(fp)

list_of_pairs = get_xy_pairs(names)
train_x, train_y, test_x, test_y = create_train_and_test_set(list_of_pairs)

for epoch in range(1, n_epochs + 1):
    train_loss = train(train_x, train_y)
    test_loss, y_pred = test(test_x, test_y)
    # print('epoch', epoch, ' train loss:', '%.5f'%(train_loss), '  test loss:', '%.5f'%(test_loss))
    y_predicted.append(y_pred)
    all_losses_train.append(train_loss)
    all_losses_test.append(test_loss)

#saving your model
torch.save(rnn, 'rnn.pt')

#Loss vs Epoch Plot
import matplotlib.pyplot as plt
plt.plot(range(1, n_epochs + 1), all_losses_train, label = 'Train Loss')
plt.plot(range(1, n_epochs + 1), all_losses_test, label = 'Test Loss')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

#Confusion Matrix Plot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
y_prediction = [torch.topk(i[0],1)[1][0] for i in y_pred]
cm = confusion_matrix(test_y, y_prediction)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.show()
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)

Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(test_y, y_prediction)

-------------------------------------------------------------------------
## RNN with SGD + Stratified train-test-split ###
n_letters=58
n_categories=18
n_hidden = 128
n_epochs = 100
print_every = 5000
plot_every = 1000
learning_rate = 0.002 # If you set this too high, it might explode. If too low, it might not learn

class RNN_SGD(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_shape (int): size of the 1-hot embeddings for each character (this will be 58)
            hidden_layer_width (int): number of nodes in the single hidden layer within the model
            n_classes (int): number of output classes
        """
        super(RNN_SGD, self).__init__()
        ### TODO Implement the network architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_layer = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.output_layer = torch.nn.Linear(input_size + hidden_size, output_size)
        # self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, input, hidden):
        """Forward function accepts tensor of input data, returns tensor of output data.
        Modules defined in constructor are used, along with arbitrary operators on tensors
        """
        
        ### TODO Implement the forward function
        comb = torch.cat((input, hidden), 1)
        hidden = self.hidden_layer(comb)
        output = self.output_layer(comb)
        output = torch.nn.functional.softmax(output, dim=1) #+ 1e-8

        #your function will return the output y(t) and hidden h(t) from equation 1 in the docs
        return output, hidden 

    def initHidden(self):
        """
        This function initializes the first hidden state of the RNN as a zero tensor.
        """
        return torch.zeros(1, self.hidden_size)

def get_xy_pairs(names):
    #TODO 
    #process the names dict and convert into a list of (x,y) pairs. x is a 1-hot tensor of size (num_characters_in_name, 1, n_letters)
    #y is a scalar representing the category of the language, there are 18 languages, assign an index between 0-17 to each language and y represents this index.
    #you may make use of the nameToTensor() function in the utils.py file to help you with this function
    list_of_pairs = []
    y = 0
    for key, val in names.items():
        for word in val:
            list_of_pairs.append((nameToTensor(word), torch.tensor(y)))
        y += 1
    return list_of_pairs

def create_train_and_test_set(list_of_pairs):
    #TODO 
    #process the list of (x,y) pairs and split them 80-20 into train and test set
    #train_x is a list of name embeddings each of size (num_characters_in_name, 1, n_letters), train_y is the correponding list of language category index. Same for test_x and test_y
    random.seed(4)
    random.shuffle(list_of_pairs)
    l = len(list_of_pairs)
    train = list_of_pairs[:int(l * 0.8)]
    test = list_of_pairs[int(l * 0.8):]
    train_x, train_y, test_x, test_y = [], [], [], []
    for pair in train:
        train_x.append(pair[0])
        train_y.append(pair[1])
    
    for pair in test:
        test_x.append(pair[0])
        test_y.append(pair[1])
    return train_x, train_y, test_x, test_y

rnn_sgd = RNN_SGD(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn_sgd.parameters(), lr=learning_rate)
criterion = torch.nn.NLLLoss()

def train(train_x, train_y):
    """train_x and train_y are lists with names and correspoonding labels"""
    loss = 0
    cnt = 0
    current_loss = 0
    for x, y in zip(train_x, train_y):
        hidden = rnn_sgd.initHidden()
        for i in range(x.size()[0]):
            output, hidden = rnn_sgd(x[i], hidden)
        loss = criterion(torch.log(output), y.unsqueeze(0)) #the unsqueeze converts the scalar y to a 1D tensor
        current_loss += loss.item()
        cnt += 1
        # pdb.set_trace()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    return current_loss/cnt

def test(train_x, train_y):
    """train_x and train_y are lists with names and correspoonding labels"""
    loss = 0
    cnt = 0
    y_prediction = []
    with torch.no_grad(): 
        for x, y in zip(train_x, train_y):
            hidden = rnn_sgd.initHidden()
            for i in range(x.size()[0]):
                output, hidden = rnn_sgd(x[i], hidden)
            y_pred = torch.log(output)
            loss += criterion(y_pred, y.unsqueeze(0)) #the unsqueeze converts the scalar y to a 1D tensor
            y_prediction.append(y_pred)
            cnt += 1
            # pdb.set_trace()
    return loss.item()/cnt, y_prediction

def create_train_and_test_set_stratify(list_of_pairs):
    #TODO 
    #process the list of (x,y) pairs and split them 80-20 into train and test set
    #train_x is a list of name embeddings each of size (num_characters_in_name, 1, n_letters), train_y is the correponding list of language category index. Same for test_x and test_y
    from sklearn.model_selection import train_test_split 
    x = [pair[0] for pair in list_of_pairs]
    y = [pair[1] for pair in list_of_pairs]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
    return train_x, train_y, test_x, test_y


# Keep track of losses for plotting
all_losses_train = []
all_losses_test = []
y_predicted = []

#names is your dataset in python dictionary form. Keys are languages and values are list of words belonging to that language
with open('names.json', 'r') as fp:
    names = json.load(fp)

list_of_pairs = get_xy_pairs(names)
train_x, train_y, test_x, test_y = create_train_and_test_set_stratify(list_of_pairs)

for epoch in range(1, n_epochs + 1):
    train_loss = train(train_x, train_y)
    test_loss, y_pred = test(test_x, test_y)
    # print('epoch', epoch, ' train loss:', '%.5f'%(train_loss), '  test loss:', '%.5f'%(test_loss))
    y_predicted.append(y_pred)
    all_losses_train.append(train_loss)
    all_losses_test.append(test_loss)

#saving your model
torch.save(rnn_sgd, 'rnn_sgd_stratify.pt')


Loss vs Epoch Plot
import matplotlib.pyplot as plt
plt.plot(range(1, n_epochs + 1), all_losses_train, label = 'Train Loss')
plt.plot(range(1, n_epochs + 1), all_losses_test, label = 'Test Loss')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

Confusion Matrix Plot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
y_prediction = [torch.topk(i[0],1)[1][0] for i in y_pred]
cm = confusion_matrix(test_y, y_prediction)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.show()
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)

Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(test_y, y_prediction)
