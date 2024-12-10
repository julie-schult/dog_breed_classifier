import sys
import os
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

sys.path.append(os.path.abspath('../preprocessing'))
from CustomDataset import CustomDataset

class MobileNet1(nn.Module):
    # INIT
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=True)
    # FORWARD PASS
    def forward(self, input_values):
        x = self.model(input_values)
        return x
    # TRAIN
    def fit(self, train_loader, max_epochs=10, lr=0.001, folder='../results'):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        train_losses = []
        train_accs = []
        epochs = [i for i in range(1, max_epochs+1)]

        for epoch in max_epochs:
            self.train()
            train_loss = 0.0
            train_acc = 0.0
            for inputs, labels in train_loader:
                # Forward propagation
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                # Backward propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            train_acc = 100 * correct / total
            train_loss = train_loss / len(train_loader.dataset)

            train_losses.append(train_loss)
            train_accs.append(train_acc)

            #Save weights
            file = os.path.join(folder, f'weights_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict' : self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, file)
        # Plots
        plt.plot(train_losses, label='Train loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss curves of training')
        plt.legend()
        file = os.path.join(folder, f'losscurves.png')
        plt.savefig(file)
        plt.close()
        # Save results
        training_data = pd.DataFrame({'epoch': epochs, 'training_loss': train_losses, 'training_acc': train_accs})
        file = os.path.join(folder, f'fit_results.csv')
        training_data.to_csv(file, index=False)
        
    # TEST
    def test(self, test_loader, weight, folder='../results'):
        accuracy = 0.0
        precision = 0.0
        sensitivity = 0.0
        specificity = 0.0
        recall = 0.0

        checkpoint = torch.load(weight)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()

