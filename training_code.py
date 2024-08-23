# Standard Libraries
import os
import random
import json
import warnings
from pathlib import Path

# Data Manipulation
import numpy as np
import pandas as pd

# Deep Learning Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.transforms import v2

# Image Processing
from PIL import Image

# Machine Learning Metrics
from sklearn.metrics import accuracy_score, f1_score

# Progress Bar
from tqdm import tqdm


# Indicate where the data is coming from - Error codes extracted earlier
file_path = r"C:\Users\Franek\Documents\python code\gemma\dataloader_error_codes.csv"
dataset_path = Path(r"C:\Users\Franek\Documents\python code\gemma\cholec80\frames")
data = pd.read_csv(file_path)

# Code to maintain reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
warnings.filterwarnings('ignore')

# Select which videos are used for training and which for testing
videos = data['video'].unique()
test_videos = [31, 64] # for annotator 4, experiment 1,2 [64, 65], 3 [31,65] # for annotator 7: [20, 74] 
train_videos = [video for video in videos if video not in test_videos]
data_train = data[data['video'].isin(train_videos)]
data_test = data[data['video'].isin(test_videos)]

# Create classes for the dataset and the model
class CholecError(Dataset):
    def __init__(self, image_folder, images_metadata, spread, transform_sequence):
        self.image_paths = image_folder
        self.images_meta = images_metadata
        self.spread = spread
        self.transform_sequence = transform_sequence

    def __len__(self):
        return len(self.images_meta)
    
    def __getitem__(self, idx):
        # Get image data
        image_data = self.images_meta.iloc[idx]

        # Get video and img idx
        video = 'video' + "{:02}".format(image_data['video'])
        idx = image_data['idx']

        # Get label
        label = float(image_data['label'])

        # Prepare to store the tensors from each image
        image_tensors = []

        # Iterate through the specified range of indices
        for i in range(idx-self.spread, idx+self.spread+1):
            idx_str = "{:06}".format(i)
            img_str = f"{video}_{idx_str}.png"
            img_path = dataset_path / video / img_str
            
            # Open and transform the image
            img = Image.open(img_path)
            img_tensor = self.transform_sequence(img)
            
            # Append the transformed image tensor
            image_tensors.append(img_tensor)

        # Stack all tensors along a new dimension
        image_tensor = torch.stack(image_tensors)

        return image_tensor, label
    
class ResNet50_BiLSTM_Binary(nn.Module):
    def __init__(self, num_frames):
        super(ResNet50_BiLSTM_Binary, self).__init__()
        # Use a pre-trained ResNet-50 and remove the fully connected layer
        resnet50 = models.resnet50(pretrained=True)
        self.resnet50_feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        
        # LSTM settings
        self.hidden_size = 512
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size=2048,
                            dropout=0.15,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=True)
        
        # Binary classification layer
        self.fc = nn.Linear(2 * self.hidden_size, 1)

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.size()
        x = x.view(batch_size * num_frames, C, H, W)
        features = self.resnet50_feature_extractor(x)
        features = features.view(batch_size, num_frames, -1)
        lstm_out, _ = self.lstm(features)
        middle_output = lstm_out[:, int(num_frames/2)+1, :]  # Select the middle frame output
        out = self.fc(middle_output)
        return out

# Set up the training routine
# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
model = ResNet50_BiLSTM_Binary(15)
model = model.to(device)

# Hyperparameters
epochs = 10
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4) # Experiment 3, 1e-5
batch_size = 4
shuffle = True
pin_memory = True


transform_sequence = v2.Compose([
    v2.Resize(size=(224, 224)),
    v2.ToTensor()])

train_dataset = CholecError(dataset_path, data_train, 7, transform_sequence)
test_dataset = CholecError(dataset_path, data_test, 7, transform_sequence)

train_dataloader = DataLoader(  train_dataset,
                                batch_size = batch_size,
                                shuffle = shuffle,
                                pin_memory = pin_memory)

test_dataloader = DataLoader(   test_dataset,
                                batch_size = batch_size,
                                shuffle = shuffle,
                                pin_memory = pin_memory)

# Model training and evaluation
results = {}
for epoch in range(epochs):
    results[f"Epoch{epoch+1}"] = {}
    print(f"Epoch: {epoch+1}")

    ### TRAIN LOOP
    train_true_labels = []
    train_predicted_probs = []
    train_loss = 0.0

    model.train()
    for images, labels in tqdm(train_dataloader):

        images, labels = images.to(device), labels.float().to(device)        

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        outputs = outputs.squeeze()

        # Compute loss
        loss = loss_fn(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_true_labels.extend(labels.cpu().numpy())
        train_predicted_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())

    # Save the results into a dictionary
    train_output_true = np.array(train_true_labels)
    train_output_predicted = np.round(train_predicted_probs)
    results[f"Epoch{epoch+1}"]['train_true'] = train_output_true
    results[f"Epoch{epoch+1}"]['train_pred'] = train_output_predicted
    results[f"Epoch{epoch+1}"]['train_loss'] = train_loss


    ### TEST LOOP
    test_true_labels = []
    test_predicted_probs = []
    test_loss = 0.0

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader):

            images, labels = images.to(device), labels.float().to(device)

            # Forward pass
            outputs = model(images)
            outputs = outputs.squeeze()
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()

            # Store for eval
            test_true_labels.extend(labels.cpu().numpy())
            test_predicted_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())

    # Save the results into dictionary
    test_output_true = np.array(test_true_labels)
    test_output_predicted = np.round(test_predicted_probs)

    accuracy = accuracy_score(test_output_true, test_output_predicted)
    f1 = f1_score(test_output_true, test_output_predicted)
    print("Accuracy:", round(accuracy, 4))
    print("F1 Score:", round(f1, 4), '\n\n')

    results[f"Epoch{epoch+1}"]['test_true'] = test_output_true
    results[f"Epoch{epoch+1}"]['test_pred'] = test_output_predicted
    results[f"Epoch{epoch+1}"]['test_loss'] = test_loss

# Save the dictionary into json file
for epoch, metrics in results.items():
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics[key] = value.tolist()

# Remember to change the name of the file depending on the experiment             
with open('cnn_bilstm_v0_4.json', 'w') as file:
    json.dump(results, file, indent=4)
