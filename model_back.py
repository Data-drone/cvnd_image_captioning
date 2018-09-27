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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        self.lstm.flatten_parameters()
        lstm_out, hidden = self.lstm(inputs)
        outputs = self.linear(lstm_out)
        
        return outputs
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        input_state = states
        sentence = []
        print("initial input: {0}".format(inputs.shape) )
        for i in range(max_len):
            lstm_out, input_state = self.lstm(inputs, input_state)
            print("hidden state length: {0}, shape of 0: {1}".format(len(input_state), input_state[0].shape))
            output = self.linear(lstm_out)
            #print(output.shape)
            max_prob, pred_index = output.max(2)
            #print(max_prob)
            word_idx = pred_index.item()
            sentence.append(word_idx)
            inputs = self.embed(pred_index) #.unsqueeze(1)
            print("next step inputs: {0}".format(inputs.shape))
        
        return sentence