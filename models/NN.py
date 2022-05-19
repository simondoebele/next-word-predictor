from datetime import datetime
import argparse
import random
import pickle
import codecs
import json
import os
import nltk
import torch
import numpy as np
from pprint import pprint

import torch.nn.functional as F
from nltk import RegexpTokenizer
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# ==================== Datasets ==================== #

# Mappings between symbols and integers, and vice versa.
# They are global for all datasets.
w2i = {}
i2w = {}
# The padding symbol will be used to ensure that all tensors in a batch
# have equal length.
PADDING_SYMBOL = ' '
w2i[PADDING_SYMBOL] = 0
i2w[0] = PADDING_SYMBOL
START_SYMBOL = '<START>'
w2i[START_SYMBOL] = 1
i2w[1] = START_SYMBOL


def load_glove_embeddings(embedding_file):
    N = len(w2i)
    embeddings = [0] * N
    with codecs.open(embedding_file, 'r', 'utf-8') as f:
        for line in f:
            data = line.split()
            word = data[0].lower()
            if word not in w2i:
                w2i[word] = N
                i2w[N] = word
                N += 1
                embeddings.append(0)
            vec = [float(x) for x in data[1:]]
            D = len(vec)
            embeddings[w2i[word]] = vec
    # Add a '0' embedding for the padding symbol
    embeddings[0] = [0] * D
    # Check if there are words that did not have a ready-made Glove embedding
    # For these words, add a random vector
    for word in w2i:
        index = w2i[word]
        if embeddings[index] == 0:
            embeddings[index] = (np.random.random(D) - 0.5).tolist()
    return D, embeddings


class LMDataset(Dataset):
    def __init__(self, filename, record_symbols=True):
        self.sentence_list = []
        self.target_sentence_list = []
        tokenizer = RegexpTokenizer(r'\w+')
        # Read the datafile
        with open(filename, encoding='utf8', errors='ignore') as f:
            lines = f.read().split('\n')
            for line in lines:
                tokenized = tokenizer.tokenize(line)
                tokens = [word.lower() for word in tokenized]
                # only take sentences that have more than 1 word.
                if len(tokens) > 1:
                    for idx, w in enumerate(tokens):
                        if w not in w2i:
                            idx = len(i2w)
                            w2i[w] = idx
                            i2w[idx] = w
                    indices = [w2i[w] for w in tokens]
                    #source_sentence.append(w2i.get(w)) # else: if word not present: possibly include: w2i[UNK_SYMBOL]
                    # we predict based on the same sentence, one word at the time
                    self.sentence_list.append(indices[:-1])
                    self.target_sentence_list.append(indices[1:])
        #print("w2i: ", len(w2i)) # NOTE: this is not the final length. More words will be added due to the embeddings
        #print("i2w: ", len(i2w))

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, idx):
        return self.sentence_list[idx], self.target_sentence_list[idx]


class PadSequence:
    """
    A callable used to merge a list of samples to form a padded mini-batch of Tensor
    """
    def __call__(self, batch, pad_word=w2i[PADDING_SYMBOL]):
        source, target = zip(*batch)
        max_len = max(map(len, source)) # needs to be the same as target
        padded_source = [[b[i] if i < len(b) else pad_word for i in range(max_len)] for b in source]
        padded_target = [[l[i] if i < len(l) else pad_word for i in range(max_len)] for l in target]
        return padded_source, padded_target

# ==================== Encoder ==================== #

class EncoderRNN(nn.Module):
    """
    Batch processing of input strings.
    """

    def __init__(self, no_of_input_symbols, embeddings=None, embedding_size=16, hidden_size=25,
                 encoder_bidirectional=False, device='cpu', use_gru=False, tune_embeddings=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.is_bidirectional = encoder_bidirectional
        self.embedding = nn.Embedding(no_of_input_symbols, embedding_size)
        self.output = nn.Linear(hidden_size, no_of_input_symbols) # no_of_input_symbols is the same as no_of_output_symbols
        if embeddings != None:
            self.embedding.weight = nn.Parameter(torch.tensor(embeddings, dtype=torch.float),
                                                 requires_grad=tune_embeddings)
        if use_gru:
            self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=self.is_bidirectional)
        else:
            self.rnn = nn.RNN(embedding_size, hidden_size, batch_first=True, bidirectional=self.is_bidirectional)
        self.device = device
        self.to(device)

    def set_embeddings(self, embeddings):
        self.embedding.weight = torch.tensor(embeddings, dtype=torch.float)

    def forward(self, x):
        # Plan for Word predictor RNN: see https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        # e_t = E * x_t
        # h_t = g(U * h_(t-1) + W * e_t)
        # y_t = softmax(V * h_t)
        indices = torch.LongTensor(x)
        # print("x-to-tensor: ", indices.size()) # batch-size x seq-length
        word_embeddings = self.embedding(indices)
        # print("word_embeddings: ", word_embeddings.size()) # torch.Size([4, 1, 50]) <-> B, T, embedding_size
        outputs, final_hidden_state = self.rnn(word_embeddings)
        #print("outputs: ", outputs.size()) # torch.Size([4, 1, 100])
        #print("final_hidden_state: ", final_hidden_state.size())
        predictions = self.output(torch.squeeze(final_hidden_state))
        #print("predictions: ", predictions.size()) # should be: batch_size x no-of-output-symbols
        return predictions


def evaluate(ds, encoder):
    correct_words_total, incorrect_words_total = 0, 0
    for sentence in ds:
        correct = sentence[1:]
        for idx, corr in enumerate(correct):
            # use the sentence up until the index idx to predict
            predictions, hidden = encoder([sentence[:idx]])
            _, predicted_tensor = predictions.topk(1)
            predicted_symbol = predicted_tensor.detach().item()
            if predicted_symbol == corr:
                correct_words_total += 1
            else:
                incorrect_words_total += 1
    all_words = correct_words_total + incorrect_words_total

    # print( t.table )
    print("Correctly predicted words    : ", correct_words_total)
    print("Incorrectly predicted words  : ", incorrect_words_total)
    print("Accuracy  : ", correct_words_total / all_words)
    print()



if __name__ == '__main__':

    # ==================== Main program ==================== #
    # Decode the command-line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-tr', '--train', default='l1_train.txt', help='A training file')
    parser.add_argument('-de', '--dev', default='l1_dev.txt', help='A test file')
    parser.add_argument('-te', '--test', default='l1_test.txt', help='A test file')
    parser.add_argument('-ef', '--embeddings', default='', help='A file with word embeddings')
    parser.add_argument('-et', '--tune-embeddings', action='store_true', help='Fine-tune GloVe embeddings') # TODO: currently not doing this!
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-hs', '--hidden_size', type=int, default=50, help='Size of hidden state')
    parser.add_argument('-bs', '--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-a', '--attention', action='store_true', help='Use attention weights')
    parser.add_argument('-b', '--bidirectional', action='store_true', help='The encoder is bidirectional') # TODO: necessary?
    parser.add_argument('-g', '--gru', action='store_true', help='Use GRUs instead of ordinary RNNs')
    parser.add_argument('-s', '--save', action='store_true', help='Save model')
    parser.add_argument('-l', '--load', type=str, help="The directory with encoder and decoder models to load")

    args = parser.parse_args()

    is_cuda_available = torch.cuda.is_available()
    print("Is CUDA available? {}".format(is_cuda_available))
    if is_cuda_available:
        print("Current device: {}".format(torch.cuda.get_device_name(0)))
    else:
        print('Running on CPU')
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.load:
        w2i = pickle.load(open(os.path.join(args.load, "source_w2i"), 'rb'))
        i2w = pickle.load(open(os.path.join(args.load, "source_i2w"), 'rb'))

        settings = json.load(open(os.path.join(args.load, "settings.json")))

        use_attention = settings['attention']

        encoder = EncoderRNN(
            len(i2w),
            embedding_size=settings['embedding_size'],
            hidden_size=settings['hidden_size'],
            encoder_bidirectional=settings['bidirectional'],
            use_gru=settings['use_gru'],
            tune_embeddings=settings['tune_embeddings'],
            device=device
        )


        encoder.load_state_dict(torch.load(os.path.join(args.load, "encoder.model")))

        print("Loaded model with the following settings")
        print("-" * 40)
        pprint(settings)
        print()
    else:
        # ==================== Training ==================== #
        # Reproducibility
        # Read a bit more here -- https://pytorch.org/docs/stable/notes/randomness.html
        random.seed(5719)
        np.random.seed(5719)
        torch.manual_seed(5719)
        torch.use_deterministic_algorithms(True)

        if is_cuda_available:
            torch.backends.cudnn.benchmark = False

        use_attention = args.attention

        # Read datasets
        training_dataset = LMDataset(args.train)
        dev_dataset = LMDataset(args.dev, record_symbols=False)

        print("Number of source words: ", len(i2w))
        print("Number of training lines: ", len(training_dataset))
        print()

        # If we have pre-computed word embeddings, then make sure these are used
        if args.embeddings:
            embedding_size, embeddings = load_glove_embeddings(args.embeddings)
        else:
            embedding_size = args.hidden_size
            embeddings = None

        training_loader = DataLoader(training_dataset, batch_size=args.batch_size, collate_fn=PadSequence())
        # can still set num_workers for speed and shuffle=True
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=PadSequence())

        criterion = nn.CrossEntropyLoss()
        # criterion = nn.NLLLoss()

        # print("no_input_symbols: ", len(w2i)) -> lots more words (that were not part of dataset, but are part of pretrained glove)

        encoder = EncoderRNN(
            len(i2w),
            embeddings=embeddings,
            embedding_size=embedding_size,
            hidden_size=args.hidden_size,
            encoder_bidirectional=args.bidirectional,
            tune_embeddings=args.tune_embeddings,
            use_gru=args.gru,
            device=device
        )

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)

        encoder.train()
        print(datetime.now().strftime("%H:%M:%S"), "Starting training.")

        #dataset = iter(training_loader)
        #data = dataset.next()
        #print(data)
        #print(source)
        #print(target)
        #source, target = data
        #exit()

        for epoch in range(args.epochs):
            total_loss = 0
            for sentence, target_sentence in training_loader:  # tqdm(training_loader, desc="Epoch {}".format(epoch + 1)):
                #print(sentence, target_sentence)
                #exit()
                encoder_optimizer.zero_grad()
                loss = 0
                # hidden is (D * num_layers, B, H)
                # outputs, hidden = encoder(sentence)
                # if args.bidirectional:
                #     hidden = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=1).unsqueeze(0)

                # The probability of doing teacher forcing will decrease
                # from 1 to 0 over the range of epochs.
                #teacher_forcing_ratio = 1  # - epoch/args.epochs

                # The input to the encoder in the first time step will be
                # the boundary symbol, regardless if we are using teacher
                # forcing or not.
                # idx = [w2i[START_SYMBOL] for sublist in sentence]
                # predicted_symbol = [w2i[START_SYMBOL] for sublist in sentence]

                sentence_length = len(sentence[0])
                # NOTE: we use teacher forcing!
                for i in range(sentence_length): # only go to seq_length - 1 because target sentence is words from respective next time step
                    # The targets will be the ith symbol of all the target
                    # strings. They will also be used as inputs for the next
                    # time step if we use teacher forcing.
                    correct = [sublist[i] for sublist in target_sentence]
                    input_to_predict = [sublist[:i+1] for sublist in sentence]
                    #print(input_to_predict)

                    # predict the next word by all the sentence so far.
                    predictions = encoder(input_to_predict)
                    #print(predictions.size()) # B x (seq-length=1) x hidden_size
                    #print(torch.tensor(correct).size())
                    #exit()
                    _, predicted_tensor = predictions.topk(1)  # argmax
                    predicted_symbols = predicted_tensor.squeeze().tolist()
                    # print(predicted_symbols)  # B x (seq-length=1) x hidden_size


                    loss += criterion(predictions, torch.tensor(correct).to(device))
                    #print(loss)
                    #exit()
                loss /= (sentence_length * args.batch_size)
                loss.backward()
                encoder_optimizer.step()
                total_loss += loss
                print(total_loss)
            print(datetime.now().strftime("%H:%M:%S"), "Epoch", epoch, "loss:", total_loss.detach().item())
            total_loss = 0

            if epoch % 10 == 0:
                print("Evaluating on the dev data...")
                evaluate(dev_dataset, encoder)

        # ==================== Save the model  ==================== #

        if (args.save):
            dt = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
            newdir = 'model_' + dt
            os.mkdir(newdir)
            torch.save(encoder.state_dict(), os.path.join(newdir, 'encoder.model'))
            with open(os.path.join(newdir, 'w2i'), 'wb') as f:
                pickle.dump(w2i, f)
                f.close()
            with open(os.path.join(newdir, 'i2w'), 'wb') as f:
                pickle.dump(i2w, f)
                f.close()


            settings = {
                'training_set': args.train,
                'test_set': args.test,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'hidden_size': args.hidden_size,
                'attention': args.attention,
                'bidirectional': args.bidirectional,
                'embedding_size': embedding_size,
                'use_gru': args.gru,
                'tune_embeddings': args.tune_embeddings
            }
            with open(os.path.join(newdir, 'settings.json'), 'w') as f:
                json.dump(settings, f)

    # ==================== Evaluation ==================== #

    encoder.eval()
    print("Evaluating on the test data...")

    test_dataset = LMDataset(args.test, record_symbols=False)
    print("Number of test sentences: ", len(test_dataset))
    print()

    evaluate(test_dataset, encoder)

    # ==================== User interaction ==================== #

    while (True):
        text = input("> ")
        if text == "":
            continue
        try:
            # same preprocessing as above.
            tokenizer = RegexpTokenizer(r'\w+')
            tokenized = tokenizer.tokenize(text)
            tokens = [word.lower() for word in tokenized]
            source_sentence = [w2i[w] for w in tokens]
        except KeyError:
            print("Erroneous input string")
            continue
        predictions, hidden = encoder([source_sentence])
        if encoder.is_bidirectional:
            hidden = hidden.permute((1, 0, 2)).reshape(1, -1).unsqueeze(0)

        _, predicted_tensor = predictions.topk(1)
        predicted_symbol = predicted_tensor.detach().item()
        #print(i2w[predicted_symbol])
        print(i2w[predicted_symbol].encode('utf-8').decode(), end=' ')
        print()



        # for i in target_sentence:
        #     print(target_i2w[i].encode('utf-8').decode(), end=' ')
        # print()

        # if use_attention:
        #     # Construct the attention table
        #     ap = torch.tensor(attention_probs).T
        #     if len(ap.shape) == 1:
        #         ap = ap.unsqueeze(0)
        #     attention_probs = ap.tolist()
        #
        #     for i in range(len(attention_probs)):
        #         for j in range(len(attention_probs[i])):
        #             attention_probs[i][j] = "{val:.2f}".format(val=attention_probs[i][j])
        #     for i in range(len(attention_probs)):
        #         if i < len(text):
        #             attention_probs[i].insert(0, source_i2w[source_sentence[i]])
        #         else:
        #             attention_probs[i].insert(0, ' ')
        #     first_row = ["Source/Result"]
        #     for w in target_sentence:
        #         first_row.append(target_i2w[w])
        #     attention_probs.insert(0, first_row)
        #     t = AsciiTable(attention_probs)
        #     print(t.table)
