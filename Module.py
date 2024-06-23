from torch import nn, optim
import torch
import torch.nn.init as init
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, encoder_outputs, decoder_hidden):
        seq_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.permute(1,0,2)
        decoder_hidden = decoder_hidden.repeat(1, seq_len, 1)
        energy = F.tanh(self.attn(torch.cat((encoder_outputs, decoder_hidden), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        return F.softmax(attention, dim=1)


class EncoderDecoderMaster(nn.Module):
    """The base class of models."""
    def __init__(self, input_size, hidden_size, concat_size, hidden_size_2, output_size, num_layers = 1, optimizer = 'SGD', learning_rate = 0.001, loss_function = 'MSE', l1 = 0.0, l2 = 0.0, clip_val=0, scheduler = None):
        super(EncoderDecoderMaster, self).__init__()


        self.output_size = output_size
        self.hidden_size = hidden_size
        self.attention = Attention(self.hidden_size)

        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_size, hidden_size, self.num_layers , batch_first=True, bidirectional=False)
        self.initialize_weights(self.encoder, 'Xavier', 1)
        
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.initialize_weights(self.decoder, 'Xavier', 1)

        self.linear = nn.Linear(hidden_size + hidden_size , hidden_size)


        #Additional vector to concatenate
        self.fc_concat = nn.Linear(concat_size, hidden_size_2)
        self.initialize_weights(self.fc_concat, 'He', 0)

        self.fc = nn.Linear(hidden_size + hidden_size_2, output_size) 
        self.initialize_weights(self.fc, 'Normal', 0)
        
        self.learning_rate = learning_rate
        self.l1_rate = l1
        self.l2_rate = l2

        self.optimizer = self.get_optimizer(optimizer, self.learning_rate, self.l2_rate)
        self.loss = self.get_loss(loss_function)
        self.metric = self.get_metric()
        self.clip_val = clip_val
        self.scheduler = self.get_scheduler(scheduler, self.optimizer)
        
        self.activation = nn.PReLU()  


    def forward(self, x, additional):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(x, (h0, c0))

        # Decoder
        decoder_input, decoder_hidden, decoder_cell = encoder_output[:,-1,:], encoder_hidden, encoder_cell

        for t in range(x.size(1)):  # Iterate over time steps
            context_vector = self.attention(encoder_output, decoder_hidden) 
            context_vector = context_vector.unsqueeze(1)  # Add a time dimension for broadcasting
            weighted_context = torch.bmm(context_vector, encoder_output)  # Weighted sum of encoder outputs
            
            decoder_input = self.linear(torch.cat((decoder_input, weighted_context.squeeze(1)), dim=1)) 
            
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input.unsqueeze(1), (decoder_hidden, decoder_cell))
            decoder_input = decoder_output[:,-1,:]
        final_hidden_state = decoder_hidden[-1]
        # Apply linear transformation to additional vector
        additional_transformed = self.activation(self.fc_concat(additional))
        # Concatenate LSTM output and additional vector
        concatenated = torch.cat((final_hidden_state, additional_transformed), dim=1)
        
        # Fully connected layer
        output = self.fc(concatenated)

        return output


    def get_optimizer(self, optimizer, learning_rate, l2):
        Optimizers = {'Adam':optim.Adam(self.parameters(), lr=learning_rate, weight_decay = l2), 'SGD': optim.SGD(self.parameters(), lr=learning_rate, momentum=0.09, weight_decay = l2)}
        return Optimizers[optimizer]

    def l1_regularization(self, loss):
        l1_reg = sum(p.abs().sum() * self.l1_rate for p in self.parameters())
        loss += l1_reg
        return loss

    def get_loss(self,loss_function):
        Loss_Functions = {'CEL': nn.CrossEntropyLoss(), 'MSE': nn.MSELoss(), 'MAE' : nn.L1Loss(), 'Huber' : nn.HuberLoss()}
        return Loss_Functions[loss_function]

    def get_metric(self):
        return torch.nn.L1Loss()

    def get_scheduler(self, scheduler, optimizer):
        if scheduler is None:
            return None
        schedulers = {'OnPlateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.15, patience=4, threshold=0.01)}
        return schedulers[scheduler]

    def xavier_init(self,tensor):
        return init.xavier_uniform_(tensor)

    def uniform_init(self,tensor):
        return init.uniform_(tensor, a=-0.1, b=0.1)

    def normal_init(self,tensor):
        return init.normal_(tensor, mean=0, std=0.01)

    def he_init(self,tensor):
        return init.kaiming_normal_(tensor, mode='fan_in', nonlinearity='relu')

    def initialize_weights(self, layer_init = None, initialisation = 'Normal', bias = 0):
        init_methods = {'Xavier': self.xavier_init, 'Uniform': self.uniform_init, 'Normal': self.normal_init, 'He': self.he_init}


        self._init_weights = init_methods[initialisation]

        if layer_init is None:
            print('no layer specified')
            parameters = self.named_parameters()
            print(f'{initialisation} initialization for all weights')
        else: 
            parameters = layer_init.named_parameters()
            print(f'{initialisation} initialization for {layer_init}')
        for name, param in parameters:
            if 'weight' in name:
                self._init_weights(param)
            elif 'bias' in name:
                # Initialize biases to 1 
                nn.init.constant_(param, bias)
        
    def clip_gradients(self, clip_value):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-clip_value, clip_value)

