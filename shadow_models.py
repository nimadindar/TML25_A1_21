import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from tqdm import tqdm

import os
import csv

class ShadowModels:
    def __init__(self, lr, num_epochs, weight_decay):
        
        self.model = models.resnet18(weights = 'IMAGENET1K_V1')
        # self.num_shadow_models = num_shadow_models
        self.lr = lr
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.num_classes = 44
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.csv_file = "./results/shadow_models_metrics.csv"

        last_layer = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(last_layer, self.num_classes)

        self.model = self.model.to(self.device)

    def train(self, model_idx, dataloader):
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr= self.lr, weight_decay=self.weight_decay)

        file_exists = os.path.exists(self.csv_file)
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['model_id', 'epoch', 'loss', 'accuracy'])

        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            progress_bar = tqdm(dataloader, desc = f"Epoch {epoch+1}/{self.num_epochs}", unit = "batch")

            for batch in progress_bar:
                _, images, labels, _ = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            epoch_loss = running_loss / len(dataloader)
            epoch_accuracy = 100 * correct / total
            
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
            self._write_to_csv(model_idx, epoch, epoch_loss, epoch_accuracy)

        torch.save(self.model.state_dict(), f"./results/shadow_model_{model_idx}_final.pth")

        return self.model
    
    def _write_to_csv(self, index, epoch, epoch_loss, epoch_acc):
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([index+1, epoch+1, f"{epoch_loss:.4f}", f"{epoch_acc:.2f}"])


