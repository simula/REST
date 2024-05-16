import torch
import torch.nn as nn 
from pathlib import Path
import os 
import pandas  as pd 
import numpy as np

def normalise(series):
    return (series-series.mean())/series.std()

def transform_to_xy_tensor(X, window_size):
    X_data = []

    for i in range(len(X) - window_size):
        X_data.append(X[i:i + window_size])

    # Convert lists of arrays to single numpy arrays
    X_data = np.array(X_data)
    return torch.tensor(X_data).float()


class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_layers: int, temperature: float):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.temperature = temperature

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 256)  # Adjusted to 256 units to match BayesianNet
        self.bn1 = nn.BatchNorm1d(256)  # Added BatchNorm
        self.dropout1 = nn.Dropout(p=dropout)  # Renamed to dropout1 for consistency

        self.fc2 = nn.Linear(256, 128)  # Adjusted to 128 units to match BayesianNet
        self.bn2 = nn.BatchNorm1d(128)  # Added BatchNorm
        self.dropout2 = nn.Dropout(p=dropout)  # Renamed to dropout2 for consistency

        self.fc3 = nn.Linear(128, 64)  # Adjusted to 64 units to match BayesianNet
        self.bn3 = nn.BatchNorm1d(64)  # Added BatchNorm
        self.dropout3 = nn.Dropout(p=dropout)  # Renamed to dropout3 for consistency

        self.fc4 = nn.Linear(64, 1)  # Added an additional fc layer to match BayesianNet

    def forward(self, x):
        # LSTM layer expects input of shape (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)

        # Only take the output from the final timestep
        x = lstm_out[:, -1, :]

        # Fully connected layers with Batch Normalization and Dropout
        x = torch.relu(self.bn1(self.fc1(x)))

        x = torch.relu(self.bn2(self.fc2(x)))

        x = torch.relu(self.bn3(self.fc3(x)))

        return torch.sigmoid(self.fc4(x) / self.temperature)

def main():
    path_to_store = Path(__file__).parent / "data" / "annotations"
    path_to_save_res = Path(__file__).parent / "data"

    model = LSTM(input_dim=1, hidden_dim=25, dropout=0.5, num_layers=2, temperature=2)
    model.load_state_dict(torch.load(path_to_save_res / 'final_model.pth', map_location=torch.device('cpu')))
    model.eval() 
    window_size = 50
    
    for i, file_name in enumerate(os.listdir(path_to_store)):
        if "60" in file_name:
            file = pd.read_csv(path_to_store / file_name, encoding="utf-8")
            file["vm"] = np.sqrt(file['x']**2 + file['y']**2 + file['z']**2)
            X = transform_to_xy_tensor(file.vm, window_size)
            X = normalise(X)
            with torch.no_grad():
                X = X.unsqueeze(2)
                outputs = model(X)
                predictions = np.where(outputs.cpu().numpy().round() == 0, 's', 'w')
            nans = np.empty((window_size, 1))
            nans[:] = np.nan
            file["LSTM"] = np.concatenate((nans, predictions), axis=0)
            file.loc[file['Sadeh'] == 'n', 'LSTM'] = 'n'
            file_name = file_name.split(".")[0]
            file.to_csv(path_to_save_res/ "pseudo_labels"/ f"{file_name}_60_secs.csv")
            print(f"Store {file_name}")
if __name__ == "__main__":
    main()