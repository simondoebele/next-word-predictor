from collections import defaultdict
import pickle
import json
import torch
import os
from RNN import EncoderRNN

class RNN(EncoderRNN):

    def __init__(self, w2i, i2w, no_of_input_symbols, embeddings=None, embedding_size=16, hidden_size=25, encoder_bidirectional=False, device='cpu', use_gru=False, tune_embeddings=False):
        super().__init__(no_of_input_symbols, embeddings, embedding_size, hidden_size, encoder_bidirectional, device, use_gru, tune_embeddings)
        self.w2i = w2i
        self.i2w = i2w
        self.no_of_input_symbols = no_of_input_symbols

    def predict_next_word(self, words: tuple, first_characters: str = '', limit: int = 4):

        source_sentence = [self.w2i[w] for w in words]

        try:
            predictions = self([source_sentence])

            if self.is_bidirectional:
                hidden = hidden.permute((1, 0, 2)).reshape(1, -1).unsqueeze(0)

            _, predicted_tensor = predictions.topk(10000)
            
            if len(predicted_tensor.shape) == 1:
                predicted_symbol = [x.item() for x in predicted_tensor]
            else:
                predicted_symbol = [x.item() for x in predicted_tensor[-1]]
            
            predictions = [self.i2w[word].encode('utf-8').decode() for word in predicted_symbol]
            predictions.remove(' ')

            filtered = list(
                filter(
                    lambda x: x.startswith(first_characters),
                    predictions
                )
            )

            return filtered[:limit]

        except: return []


def load_RNN(folder_name: str):

    w2i = pickle.load(open(os.path.join(folder_name, "w2i"), 'rb'))
    w2i = defaultdict(int, w2i)
    i2w = pickle.load(open(os.path.join(folder_name, "i2w"), 'rb'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    settings = json.load(open(os.path.join(folder_name, "settings.json")))
    
    use_attention = settings['attention']

    encoder = RNN(
        w2i,
        i2w,
        len(i2w),
        embedding_size=settings['embedding_size'],
        hidden_size=settings['hidden_size'],
        encoder_bidirectional=settings['bidirectional'],
        use_gru=settings['use_gru'],
        tune_embeddings=settings['tune_embeddings'],
        device=device
    )

    encoder.load_state_dict(torch.load(os.path.join(folder_name, "encoder.model")))

    return encoder

# model = load_RNN('model_2022-05-19_16_15_44_320110')
# print(model.predict_next_word(('hello', 'how'), 'th'))