#!/usr/bin/python3
# DIPEN RUPANI 112321338
# CSE354, Spring 2021
##########################################################
## a2_Rupani_112321338.py

# import os
import sys
# import json #for reading json encoded files into opjects
import re
import numpy as np
import torch
import torch.nn as nn

# np.set_printoptions(threshold=sys.maxsize)

sys.stdout = open('a2_Rupani_112321338_OUTPUT.txt', 'w') # EDIT THIS

# loads data into a list from given filename
def loadData(filename):
    data = []
    with open(filename) as f:
        line = f.readline()
        while line:
            data.append(line.split('\t', 3)) # split by tab but only 3 times for lemma, sense, context
            line = f.readline()
    return data

# converts sense from string to int eg. string(process%01:09:00::) -> int(09)
def convertSense(data):
    sense_counter = 0
    senses = {}
    for d in data:
        if d[1] in senses:
            d[1] = senses[d[1]]
        else:
            sense_counter+=1
            senses[d[1]] = sense_counter
            d[1] = senses[d[1]]
    return data

# receives a list of words and returns the same set of words back with 
    # <head> removed from target word and the index of head
def locateHead(words):
    headMatch=re.compile(r'<head>([^<]+)') # matches contents of head  
    for i in range(len(words)):
            m = headMatch.match(words[i])
            if m: #a match: we are at the target token
                words[i] = words[i].split('>')[1]
                return (words, i)

# creates a vocabulary with given data
    # split the data up into tokens, keep track of index of head. 
    # remove lemma/POS and add index of head as a feature to the index
def createVocab(data, vocab = {}):
    for d in data:
        words = [(wlp.split('/')[0]).lower() for wlp in d[2].split()]
        words, head_index = locateHead(words)
        d.append(head_index)
        d[2] = " ".join(words)
        for word in words:
            vocab[word] = vocab.get(word, 0) + 1

    sorted_vocab = {}
    sorted_word_count = sorted(vocab, key = lambda k: (vocab[k], k))
    for w in sorted_word_count:
        sorted_vocab[w] = vocab[w]
        
    final_vocab = {k: sorted_vocab[k] for k in list(sorted_vocab)[-2000:]}
    temp_vocab = {k: sorted_vocab[k] for k in list(sorted_vocab)[:-2000]}
    final_vocab['OOV'] = sum(temp_vocab.values())

    vocab = list(final_vocab.keys())

    return vocab

# accepts data and vocab and returns X = 2 1-hot encodings.
def createOneHot(data, vocab):
    for d in data:
        word_list = d[2].split(' ')

        before_word = word_list[(d[3]-1) if ((d[3]-1) > 0) else 0]
        if before_word == word_list[0]:
            before_onehot = [0 for k in vocab]
        else:
            before_onehot = [0 if k != before_word else 1 for k in vocab]
        if 1 not in before_onehot and before_word != word_list[0]:
            before_onehot[-1] = 1

        after_word = word_list[(d[3]+1) if ((d[3]+1) <= (len(word_list)-1)) else (len(word_list)-1)]
        if after_word == word_list[len(word_list)-1]:
            after_onehot = [0 for k in vocab]
        else:   
            after_onehot = [0 if k != after_word else 1 for k in vocab]
        if 1 not in after_onehot and after_word != word_list[len(word_list)-1]:
            after_onehot[-1] = 1

        d.append(before_onehot + after_onehot)

# creates a cooccurance matrix given a vocabulary and context. 
def createCooccuranceMatrix(vocab, context):
    cooccur_matrix = np.zeros((len(vocab), len(vocab))) # creates a list of lists for matrix

    vocab_dict = {vocab[w]: w for w in range(len(vocab))}
    for c in context: 
        words = c.split()
        for index1 in range(len(words)):
            for index2 in range(len(words)):
                if index1 != index2:
                    cooccur_matrix[vocab_dict.get(words[index1], -1)][vocab_dict.get(words[index2], -1)] += 1

    return cooccur_matrix

# prep data for next part, change all oov words to 'OOV'
def cooccurMatrixDataPrep(data, vocab):
    context = []
    for d in data:
        words = d[2].split()
        for w in range(len(words)):
            if words[w] not in vocab:
                words[w] = 'OOV'
        d[2] = " ".join(words)
        context.append(d[2])
    return context

# create the embedding features given the dict and the contexts
def createEmbeddingsFeature(data, embeddings_dict):
    X = []
    for d in data:
        context = d[2].split()
        try:
            word_two_before = embeddings_dict[context[d[3]-2]].tolist()
            # print(f"{word_two_before.shape}")
        except IndexError: 
            word_two_before = torch.zeros(50).tolist()
            # print(f"default: {word_two_before.shape}")
        try:
            word_two_after = embeddings_dict[context[d[3]+2]].tolist()
            # print(f"{word_two_after.shape}")
        except IndexError: 
            word_two_after = torch.zeros(50).tolist()
            # print(f"default: {word_two_after.shape}")
        try:
            word_one_before = embeddings_dict[context[d[3]-1]].tolist()
            # print(f"{word_one_before.shape}")
        except IndexError: 
            word_one_before = torch.zeros(50).tolist()
            # print(f"default: {word_one_before.shape}")
        try:
            word_one_after = embeddings_dict[context[d[3]+1]].tolist()
            # print(f"{word_one_after.shape}")
        except IndexError: 
            word_one_after = torch.zeros(50).tolist()
            # print(f"default: {word_one_after.shape}")

        # new_emb = torch.cat((word_two_before, word_one_before, word_one_after, word_two_after), 0) 
        X.append(word_two_before + word_one_before + word_one_after + word_two_after)
    return np.array(X)

# train and evaluate on test set - for multi class logistic regression classifier. 
def trainAndEvaluate(X_train, X_test, y_train, y_test, index_data, learning_rate = 0.01, epochs = 1000):
    print("\nTraining Logistic Regression...")
    model = LogReg(X_train.shape[1], max(y_train.tolist())+1)
    sgd = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    #training loop:
    for i in range(epochs):
        model.train()
        sgd.zero_grad()

        #forward pass:
        y_pred = model(X_train)
        loss = loss_func(y_pred, y_train)

        #backward:
        loss.backward()
        sgd.step()
        if i % 20 == 0:
            print("  epoch: %d, loss: %.5f" %(i, loss.item()))

    #calculate accuracy on test set:
    with torch.no_grad():
        ytest_pred = model(X_test)

        print(f"{index_data[0][0].split('.')[0]}\npredictions for {index_data[0][0]}: {ytest_pred[0]}\npredictions for {index_data[1][0]}: {ytest_pred[1]}")
        predictions = [pred.tolist().index(max(pred)) for pred in ytest_pred]
        correct = [ 1 for i in range(len(y_test)) if y_test[i] == predictions[i]]
        print(f"correct: {len(correct)} out of {len(y_test)}\n")

# log reg class - for multi class logistic regression classifier
class LogReg(nn.Module):
    def __init__(self, num_feats, num_classes, learn_rate = 0.01, device = torch.device("cpu") ):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(num_feats+1, num_classes) #add 1 to features for intercept

    def forward(self, X):
        newX = torch.cat((X, torch.ones(X.shape[0], 1)), 1) #add intercept
        return self.linear(newX)

if __name__ == "__main__":
# PART 1.1 (a)--------------------------------------------------------------------
    if len(sys.argv) != 3:
        print("USAGE: python3 a2_lastname_id.py onesec_train.tsv onesec_test.tsv")
        sys.exit(1)
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    train = loadData(train_file)
    test = loadData(test_file)
    splitPoints = [807, 1614, 202, 404]

# PART 1.1 (b)--------------------------------------------------------------------
    # some overly complicated list manipulations to make up for my poor design decisions :(
    train_process, train_machine, train_language = train[:splitPoints[0]], train[splitPoints[0]:splitPoints[1]], train[splitPoints[1]:] # split by words
    test_process, test_machine, test_language = test[:splitPoints[2]], test[splitPoints[2]:splitPoints[3]], test[splitPoints[3]:] # split test set by words

    process = train_process + test_process # now group by word so that we can get all senses per word
    machine = train_machine + test_machine
    language = train_language + test_language

    process, machine, language = convertSense(process), convertSense(machine), convertSense(language)

    train = process[:splitPoints[0]] + machine[:splitPoints[0]] + train_language[:splitPoints[0]] # now split and combine again to proceed normally
    test = process[splitPoints[0]:] + machine[splitPoints[0]:] + language[splitPoints[0]:]
    
# PART 1.1 (c)--------------------------------------------------------------------
    vocab = createVocab(train)
    createVocab(test)
    
# PART 1.2------------------------------------------------------------------------
    #divide the training and testing set into process machine and language.
    # the goal here is to define 2 one-hot encoding of length of vocab.
    createOneHot(train, vocab)
    createOneHot(test, vocab)

# PART 1.3 & 1.4-------------------------------------------------------------------------
    # now we separate data into x and y test/train_process, test/train_machine, 
    # and test/train_language for each case just keep sense(data[1]) as y variable and the onehot encoding(data[3])

    X_train, X_test, y_train, y_test = \
        np.array([d[4] for d in train]), \
        np.array([d[4] for d in test]), \
        np.array([d[1]-1 for d in train]), \
        np.array([d[1]-1 for d in test]) 
    

    X_train_process, X_train_machine, X_train_language = \
        torch.from_numpy(X_train[:splitPoints[0]].astype(np.int_)), \
        torch.from_numpy(X_train[splitPoints[0]:splitPoints[1]].astype(np.int_)), \
        torch.from_numpy(X_train[splitPoints[1]:].astype(np.int_))

    X_test_process, X_test_machine, X_test_language = \
        torch.from_numpy(X_test[:splitPoints[2]].astype(np.int_)), \
        torch.from_numpy(X_test[splitPoints[2]:splitPoints[3]].astype(np.int_)), \
        torch.from_numpy(X_test[splitPoints[3]:].astype(np.int_))

    y_train_process, y_train_machine, y_train_language = \
        torch.from_numpy(y_train[:splitPoints[0]].astype(np.int_)), \
        torch.from_numpy(y_train[splitPoints[0]:splitPoints[1]].astype(np.int_)), \
        torch.from_numpy(y_train[splitPoints[1]:].astype(np.int_))

    y_test_process, y_test_machine, y_test_language = \
        torch.from_numpy(y_test[:splitPoints[2]].astype(np.int_)), \
        torch.from_numpy(y_test[splitPoints[2]:splitPoints[3]].astype(np.int_)), \
        torch.from_numpy(y_test[splitPoints[3]:].astype(np.int_))

    trainAndEvaluate(X_train_process, X_test_process, y_train_process, y_test_process, test)
    trainAndEvaluate(X_train_machine, X_test_machine, y_train_machine, y_test_machine, test[202:])
    trainAndEvaluate(X_train_language, X_test_language, y_train_language, y_test_language, test[404:])

# PART 2.1---------------------------------------------------------------------------
    # start by processing the contexts to be passed in
    context = cooccurMatrixDataPrep(train, vocab)
    cooccurMatrixDataPrep(test, vocab)
    
    # create the cooccurrance matrix
    matrix = createCooccuranceMatrix(vocab, context)

# PART 2.2----------------------------------------------------------------------------
    # standardize and get it ready for svd
    matrix = (matrix - np.mean(matrix))/np.std(matrix)
    matrix = torch.from_numpy(matrix)

    # svd
    U, D, V = torch.svd(matrix)

    # take only the first 50 features out of this to create the embeddings
    embeddings = U[:,:50]
    embeddings_dict = {vocab[ei]:embeddings[ei] for ei in range(len(embeddings))}

# PART 2.3------------------------------------------------------------------------------
    print('(language, process)')
    print(np.linalg.norm(embeddings_dict['language'] - embeddings_dict['process']))
    print('\n(machine, process)')
    print(np.linalg.norm(embeddings_dict['machine'] - embeddings_dict['process']))
    print('\n(language, speak)')
    print(np.linalg.norm(embeddings_dict['language'] - embeddings_dict['speak']))
    print('\n(word, words)')
    print(np.linalg.norm(embeddings_dict['word'] - embeddings_dict['words']))
    print('\n(word, the)')
    print(np.linalg.norm(embeddings_dict['word'] - embeddings_dict['the']))

# PART 3.1-------------------------------------------------------------------------------
    # returns a numpy feature matrix
    X_train = createEmbeddingsFeature(train, embeddings_dict)
    X_test = createEmbeddingsFeature(test, embeddings_dict)

# PART 3.2&3.3---------------------------------------------------------------------------
    #split and convert to tensors
    X_train_process, X_train_machine, X_train_language = \
        torch.from_numpy(X_train[:splitPoints[0]].astype(np.float32)), \
        torch.from_numpy(X_train[splitPoints[0]:splitPoints[1]].astype(np.float32)), \
        torch.from_numpy(X_train[splitPoints[1]:].astype(np.float32))

    X_test_process, X_test_machine, X_test_language = \
        torch.from_numpy(X_test[:splitPoints[2]].astype(np.float32)), \
        torch.from_numpy(X_test[splitPoints[2]:splitPoints[3]].astype(np.float32)), \
        torch.from_numpy(X_test[splitPoints[3]:].astype(np.float32))

    y_train_process, y_train_machine, y_train_language = \
        torch.from_numpy(y_train[:splitPoints[0]].astype(np.int_)), \
        torch.from_numpy(y_train[splitPoints[0]:splitPoints[1]].astype(np.int_)), \
        torch.from_numpy(y_train[splitPoints[1]:].astype(np.int_))

    y_test_process, y_test_machine, y_test_language = \
        torch.from_numpy(y_test[:splitPoints[2]].astype(np.int_)), \
        torch.from_numpy(y_test[splitPoints[2]:splitPoints[3]].astype(np.int_)), \
        torch.from_numpy(y_test[splitPoints[3]:].astype(np.int_))

    trainAndEvaluate(X_train_process, X_test_process, y_train_process, y_test_process, test)
    trainAndEvaluate(X_train_machine, X_test_machine, y_train_machine, y_test_machine, test[202:])
    trainAndEvaluate(X_train_language, X_test_language, y_train_language, y_test_language, test[404:])

