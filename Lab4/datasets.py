from keras import models, layers
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Bidirectional, SimpleRNN, Dense
from keras.utils import to_categorical
from conll_dictorizer import CoNLLDictorizer
import numpy as np

def load_conll2003_en():
    train_file = 'eng.train'
    dev_file = 'eng.valid'
    test_file = 'eng.test'
    column_names = ['form', 'ppos', 'pchunk', 'ner']
    train_sentences = open(train_file).read().strip()
    dev_sentences = open(dev_file).read().strip()
    test_sentences = open(test_file).read().strip()
    return train_sentences, dev_sentences, test_sentences, column_names


if __name__ == '__main__':
    OPTIMIZER = 'rmsprop'
    BATCH_SIZE = 32
    EPOCHS = 16
    EMBEDDING_DIM = 100
    MAX_SEQUENCE_LENGTH = 150
    LSTM_UNITS = 512
    print("reading")
    train_sentences, dev_sentences, test_sentences, column_names = load_conll2003_en()

    conll_dict = CoNLLDictorizer(column_names, col_sep=' +')
    train_dict = conll_dict.transform(train_sentences)
    test_dict = conll_dict.transform(test_sentences)

    file = 'glove.6B.100d.txt'
    embeddings = {}
    glove = open(file)
    for line in glove:
        values = line.strip().split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        embeddings[word] = vector
    glove.close()
    glove_dict = embeddings
    embedded_words = sorted(list(glove_dict.keys()))

    """
    result = {}
    result2 = {}
    result3 = {}
    k = 0
    print("calculating")
    for i in glove_dict:

        val = round(cosine_similarity([glove_dict['sweden'], glove_dict[i]])[0][1], 4)
        if val > 0:
            result[i] = val
        val2 = round(cosine_similarity([glove_dict['table'], glove_dict[i]])[0][1], 4)
        if val2 > 0:
            result2[i] = val2
        val3 = round(cosine_similarity([glove_dict['france'], glove_dict[i]])[0][1], 4)
        if val3 > 0:
            result3[i] = val3

    print(glove_dict['table'])
    print("sorting")
    pairlist = []
    for k, v in sorted(result.items(), key=itemgetter(1)):  
        pairlist.append(k)

    print(pairlist[-5:])

    pairlist = []
    for k, v in sorted(result2.items(), key=itemgetter(1)):
        pairlist.append(k)

    print(pairlist[-5:])

    pairlist = []
    for k, v in sorted(result3.items(), key=itemgetter(1)):
        pairlist.append(k)

    print(pairlist[-5:])
    """
    ###['austria', 'finland', 'norway', 'denmark', 'sweden']
    ###['room', 'place', 'bottom', 'tables', 'table']
    ###['spain', 'britain', 'french', 'belgium', 'france']

    def to_index(X, idx):
        """
        Convert the word lists (or POS lists) to indexes
        :param X: List of word (or POS) lists
        :param idx: word to number dictionary
        :return:
        """
        X_idx = []
        for x in X:
            # We map the unknown words to one
            x_idx = list(map(lambda x: idx.get(x, 1), x))
            X_idx += [x_idx]
        return X_idx


    def build_sequences(corpus_dict, key_x='form', key_y='ner', tolower=True):
        """
        Creates sequences from a list of dictionaries
        :param corpus_dict:
        :param key_x:
        :param key_y:
        :return:
        """
        X = []
        Y = []
        for sentence in corpus_dict:
            x = []
            y = []
            for word in sentence:
                x += [word[key_x]]
                y += [word[key_y]]
            if tolower:
                x = list(map(str.lower, x))
            X += [x]
            Y += [y]
        return X, Y


    XI,YI = build_sequences(train_dict)
    vocabulary_words = list(set([word for sentence in XI for word in sentence]))

    embeddings_words = glove_dict.keys()
    vocabulary_words = sorted(set(vocabulary_words + list(embeddings_words)))

    word_rev_idx = dict(enumerate(vocabulary_words, start=2))
    word_idx = {v: k for k, v in word_rev_idx.items()}

    ner_list = sorted(list(set([ner for sentence in YI for ner in sentence])))
    NB_CLASSES = len(ner_list)

    ner_rev_idx = dict(enumerate(ner_list, start=2))
    ner_idx = {v: k for k, v in ner_rev_idx.items()}

    X_idx = to_index(XI, word_idx)
    Y_idx = to_index(YI, ner_idx)

    X = pad_sequences(X_idx)
    Y = pad_sequences(Y_idx)

    Y_train = to_categorical(Y, num_classes=len(ner_list) + 2)

    rdstate = np.random.RandomState(1234567)
    embedding_matrix = rdstate.uniform(-0.05, 0.05,(len(vocabulary_words) + 2, 100))

    for word in vocabulary_words:
        if word in glove_dict:
            # If the words are in the embeddings, we fill them with a value
            embedding_matrix[word_idx[word]] = glove_dict[word]



    model = models.Sequential()
    model.add(layers.Embedding(len(vocabulary_words) + 2,
                               100,
                               mask_zero=True,
                               input_length=None))
    model.layers[0].set_weights([embedding_matrix])
    # The default is True
    model.layers[0].trainable = False
    # model.add(SimpleRNN(100, return_sequences=True))
    # model.add(Bidirectional(SimpleRNN(100, return_sequences=True)))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    #model.add(layers.Dense(NB_CLASSES+2, input_dim=X.shape[2], activation='softmax'))
    model.add(layers.Dropout(0.25))
    #model.add(Bidirectional(LSTM(512)))
    #model.add(layers.Dropout(0.2))

    model.add(Dense(NB_CLASSES + 2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=OPTIMIZER,
                  metrics=['acc'])
    model.summary()
    try:
        model.fit(X, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    except KeyboardInterrupt as ki:
        pass
    model.save("model.h5")

    #model = models.load_model("model.h5")
    #model.save("model2.h5")

    X_test_cat, Y_test_cat = build_sequences(test_dict)
    # We create the parallel sequences of indexes
    X_test_idx = to_index(X_test_cat, word_idx)
    Y_test_idx = to_index(Y_test_cat, ner_idx)

    X_test_padded = pad_sequences(X_test_idx)
    Y_test_padded = pad_sequences(Y_test_idx)
    # One extra symbol for 0 (padding)
    Y_test_padded_vectorized = to_categorical(Y_test_padded,
                                              num_classes=len(ner_list) + 2)
    test_loss, test_acc = model.evaluate(X_test_padded,
                                         Y_test_padded_vectorized)
    corpus_ner_predictions = model.predict(X_test_padded)

    ner_pred_num = []
    for sent_nbr, sent_ner_predictions in enumerate(corpus_ner_predictions):
        ner_pred_num += [sent_ner_predictions[-len(X_test_cat[sent_nbr]):]]

    ner_pred = []
    for sentence in ner_pred_num:
        ner_pred_idx = list(map(np.argmax, sentence))
        ner_pred_cat = list(map(ner_rev_idx.get, ner_pred_idx))
        ner_pred += [ner_pred_cat]

    filename = "output"
    file = open(filename, "w")
    total, correct, total_ukn, correct_ukn = 0, 0, 0, 0
    for id_s, sentence in enumerate(X_test_cat):
        for id_w, word in enumerate(sentence):
            file.write(word)
            file.write(" ")
            file.write("_")
            file.write(" ")
            file.write("_")
            file.write(" ")
            file.write(Y_test_cat[id_s][id_w])
            file.write(" ")
            file.write(ner_pred[id_s][id_w])
            file.write("\n")
            total += 1
            if ner_pred[id_s][id_w] == Y_test_cat[id_s][id_w]:
                correct += 1
            # The word is not in the dictionary
            if word not in word_idx:
                total_ukn += 1
                if ner_pred[id_s][id_w] == Y_test_cat[id_s][id_w]:
                    correct_ukn += 1
    file.close()
    print('total %d, correct %d, accuracy %f' %
          (total, correct, correct / total))

    if total_ukn != 0:
        print('total unknown %d, correct %d, accuracy %f' %
              (total_ukn, correct_ukn, correct_ukn / total_ukn))




