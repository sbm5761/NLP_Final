import torch.nn as nn
import torchvision.models as models
import torch
from keras.utils import plot_model
from tensorflow.keras.layers import *
import tensorflow as tf

'''
Utilized the encoder and decoder structure from the following source. Giving credit where credit is due
Sources: https://github.com/sauravraghuvanshi/Udacity-Computer-Vision-Nanodegree-Program/tree/master/project_2_image_captioning_project

'''

#Encoder of CNN: ResNET50
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

#Decoder of LSTM
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        ''' Initialize the layers of this model.'''
        super().__init__()
    
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size) 
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size):
        return (torch.zeros((1, batch_size, self.hidden_size), device="mps"), torch.zeros((1, batch_size, self.hidden_size), device="mps"))

    def forward(self, features, captions):
        """ Define the feedforward behavior of the model """
        
        # Discard the <end> word to avoid predicting when <end> is the input of the RNN
        captions = captions[:, :-1]   
        batch_size = features.shape[0] 
        self.hidden = self.init_hidden(batch_size) 
        embeddings = self.word_embeddings(captions.long()) 
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1) 
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden) 
        outputs = self.linear(lstm_out) 

        return outputs
    
    def sample(self, inputs):
        output = []
        batch_size = inputs.shape[0] 
        hidden = self.init_hidden(batch_size) 
        max_seq_length = 50

        for i in range(max_seq_length):
            lstm_out, hidden = self.lstm(inputs, hidden) 
            outputs = self.linear(lstm_out) 
            outputs = outputs.squeeze(1)
            _, max_indice = torch.max(outputs, dim=1) 
            
            output.append(max_indice.cpu().numpy()[0].item()) 

            if (max_indice == 10):
                # We predicted the <end> word, so there is no further prediction to do
                break
            
            inputs = self.word_embeddings(max_indice)
            inputs = inputs.unsqueeze(1) 
            
        return output
