import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size_1=1024, hidden_size_2=256, hidden_size_3=64, output_size=1, dropout_rate=0.6):
        super(MLPClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_3, output_size),
        )

    def forward(self, x):
        return self.network(x)
    
class LLaVAMLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size_1=1024, hidden_size_2=256, hidden_size_3=64, output_size=1, dropout_rate=0.6):
        super(LLaVAMLPClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_3, output_size),
        )

    def forward(self, x):
        return self.network(x)

