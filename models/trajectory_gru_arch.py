import torch.nn as nn

class TrajectoryGRU(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super().__init__()
        self.gru = nn.GRU(input_size = input_size,hidden_size = hidden_size, num_layers=num_layers,batch_first = True)
        self.fc1 = nn.Linear(hidden_size,64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)
        self.fc2 = nn.linear(64,output_size)

    def forward(self,x):
        output, hidden = self.gru(x)

        last_output = output[:,-1,:]
        x = self.fc1(last_output)

        x = self.relu(x)

        x = self.dropout(x)

        prediction = self.fc2(x)

        return prediction