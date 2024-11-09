import torch
import torch.nn as nn

class CNN1D_LSTM_EEG(nn.Module):
    def __init__(self, num_classes=40):
        super(CNN1D_LSTM_EEG, self).__init__()

        # Define CNN layers
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout1d(0.5)
        """self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(16)"""

        # LSTM layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Pass input through CNN layers
            
        x = self.conv1(x)
        x = self.bn1(x)  # Apply Batch Normalization after conv1
        x = torch.relu(x)
        #x = self.dropout(x)
            
        x = self.conv2(x)
        x = self.bn2(x)  # Apply Batch Normalization after conv2
        x = torch.relu(x)
        x = self.drouput(x)    
        """x = self.conv3(x)
        print(f"Shape after conv3: {x.shape}")  # Debug shape after conv3
        x = self.bn3(x)  # Apply Batch Normalization after conv3
        x = torch.relu(x)"""

        # Prepare data for LSTM (batch_size, seq_length, input_size)
        x = x.permute(0, 2, 1)  # Change to (batch_size, sequence_length, channels)

        # Pass through LSTM layers
        _, (hn, _) = self.lstm(x)

        # Use the last hidden state as the output
        x = hn[-1]

        # Fully connected layer for final classification
        x = self.fc(x)
        return x
        
            
# Model instance and parameters
#model = CNN1D_LSTM_EEG(num_classes=40)

# Model summary
#print(model)
