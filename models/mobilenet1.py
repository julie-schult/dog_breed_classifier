import sys
import os
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

sys.path.append(os.path.abspath('../preprocessing'))
from CustomDataset import CustomDataset1, CustomDataset2, CustomDataset3

class MobileNet1(nn.Module):
    # INIT
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 120)
        )
    # FORWARD PASS
    def forward(self, input_values):
        x = self.model(input_values)
        x = self.hidden_layers(x)
        x = self.classifier(x)
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

        all_paths = []
        all_preds = []
        all_labels = []

        checkpoint = torch.load(weight)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()

        for paths, inputs, labels in test_loader:
            outputs = self.forward(inputs)
            _, predicted = outputs.max(1)

            all_paths.append(paths)
            all_preds.append(predicted)
            all_labels.append(labels)

        all_paths = [item for sublist in all_paths for item in sublist]
        all_preds = [item for sublist in all_preds for item in sublist]
        all_labels = [item for sublist in all_labels for item in sublist]

        results = pd.DataFrame({
            'path': all_paths,
            'predicted': all_preds,
            'label': all_labels
        })
        file = os.path.join(folder, f'test_results.csv')
        results.to_csv(file, index=False)

        correct = sum( x == y for x,y in zip(all_labels, all_preds))
        accuracy = correct / len(all_paths)

        print(f'Accuracy: {accuracy:.5f}')

        cm = confusion_matrix(all_labels, all_preds)
        TN,FP, FN, TP = cm.ravel()
        
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        print(f'Sensitivity={sensitivity:.3f}, Specificity={specificity:.3f}, Precision={precision:.3f}, Recall={recall:.3f}')

        f = plt.figure(figsize=(8,6))
        ax= f.add_subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Greens')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'Confusion Matrix')
        ax.xaxis.set_ticklabels(['HC', 'PD'])
        ax.yaxis.set_ticklabels(['HC', 'PD'])
        file = os.path.join(folder, f'cm.png')
        f.savefig(file, dpi=400)
        plt.close(f)