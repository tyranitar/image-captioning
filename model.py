import torch
import torch.nn as nn
import torchvision.models as models


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
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderRNN, self).__init__()
        
        dropout = 0.2
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features, captions):
        features = features.view(features.shape[0], 1, features.shape[1])
        # We don't input <end>.
        embeds = self.embed(captions[:, :-1])
        x = torch.cat((features, embeds), 1)
        # Input: (batch size, captions length, embed size).
        x = self.lstm(x)[0]
        # Input: (batch size, captions length, hidden size).
        x_shape = tuple(x.shape)
        x = x.contiguous().view(-1, x_shape[2])
        x = self.fc(self.dropout(self.bn(x)))
        x = x.view(x_shape[0], x_shape[1], -1)
        
        return x
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        x, h = inputs, states
        
        for i in range(max_len):
            x, h = self.lstm(x, h)
            x_shape = tuple(x.shape)
            x = x.contiguous().view(-1, x_shape[2])
            x = self.fc(self.bn(x))
            x = x.view(x_shape[0], x_shape[1], -1)
            _, x = torch.max(x, dim=2)
            
            outputs.append(x.item())
            
            x = self.embed(x)
        
        return outputs