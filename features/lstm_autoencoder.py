import torch.nn as nn


class BehaviourLSTMPredictor(nn.Module):
    def __init__(self, input_size: int = 8, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        final_hidden = lstm_output[:, -1, :]
        prediction = self.output_layer(final_hidden)
        return prediction
    