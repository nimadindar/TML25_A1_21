import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms

from load_datasets import MembershipDataset
from shadow_models import ShadowModels
from split_data import split_dataset_balanced
from rmia_offline import rmia_offline

import os
import requests
import pandas as pd
from dotenv import load_dotenv
import numpy as np

TRAIN_SHADOW_MODELS = False
INFERENCE = False
SUBMIT_TO_SCORE_BOARD = True

NUM_SHADOW_MODELS = 2
BATCH_SIZE = 256
lr = 1e-3
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-4

mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]

transform = transforms.Compose([
    transforms.Normalize(mean=mean, std=std)
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if TRAIN_SHADOW_MODELS:
    pub_dataset : MembershipDataset = torch.load("./pub.pt", weights_only = False)
    pub_dataset.transform = transform

    try:
        disjoint_subsets = split_dataset_balanced(pub_dataset, NUM_SHADOW_MODELS)
        disjoint_dataloaders = [DataLoader(subset, batch_size = BATCH_SIZE, shuffle=True)
                                for subset in disjoint_subsets]
        
        shadow_models = ShadowModels(lr, NUM_EPOCHS, WEIGHT_DECAY)
        
        print(f"Started training {NUM_SHADOW_MODELS} shadow models...")
        for num_model in range(NUM_SHADOW_MODELS):
            print(f"Training the {num_model+1} out of {NUM_SHADOW_MODELS} shadow models...")
            shadow_models.train(num_model, disjoint_dataloaders[num_model])
        print(f"Training shadow models finished successfully!")

    except ValueError as e:
        print(f"Error occurred:{e}")

if INFERENCE:
    target_model = resnet18(weights = None)
    target_model.fc = torch.nn.Linear(512,44)
    ckpt = torch.load("./01_MIA.pt", map_location="cpu")
    target_model.load_state_dict(ckpt)
    target_model.to(device)

    shadow_models_list = []         

    for i in range(NUM_SHADOW_MODELS):
        weight_path = f"./results/shadow_model_{i}_final.pth"

        shadow_model = resnet18(weights = None)
        shadow_model.fc = torch.nn.Linear(512,44)
        ckpt_shadow = torch.load(weight_path, map_location="cpu")
        shadow_model.load_state_dict(ckpt_shadow)
        shadow_model.to(device)

        shadow_models_list.append(shadow_model)

    
    pub_dataset : MembershipDataset = torch.load("./pub.pt", weights_only = False)
    pub_dataset.transform = transform
    pub_dataloader = DataLoader(pub_dataset, batch_size = BATCH_SIZE, shuffle=False)

    priv_dataset : MembershipDataset = torch.load("./priv_out.pt", weights_only = False)
    priv_dataset.transform = transform
    priv_dataloader = DataLoader(priv_dataset, batch_size = BATCH_SIZE, shuffle=False)

    scores = rmia_offline(target_model, shadow_models_list, priv_dataloader, pub_dataloader, 
                 gamma = 2.0, a = 0.3, num_z_samples = len(pub_dataset))

    print("Scores type:", type(scores))
    print("Scores shape:", scores.shape if isinstance(scores, np.ndarray) else len(scores))
    print("Scores sample:", scores[:5] if len(scores) > 0 else scores)

    try:
        df = pd.DataFrame(
            {
                "ids": scores['id'],
                "score": scores['score'],
            }
        )
        df.to_csv("scores.csv", index=None)
        print("Scores and IDs saved to scores.csv")
    except (IndexError, TypeError) as e:
        print(f"Error creating DataFrame: {e}")
        print("Ensure rmia_offline returns a structured array with 'id' and 'score' fields.")
        # Fallback: Collect IDs manually (if using original rmia_offline)
        ids_list = []
        scores_list = scores.tolist() if isinstance(scores, np.ndarray) else scores
        for ids, _, _, _ in priv_dataloader:
            for i in range(len(ids)):
                ids_list.append(ids[i].item())
        if len(ids_list) == len(scores_list):
            df = pd.DataFrame(
                {
                    "ids": ids_list,
                    "score": scores_list,
                }
            )
            df.to_csv("scores.csv", index=None)
            print("Scores and IDs saved to scores.csv (fallback method)")
        else:
            print(f"Cannot save: {len(ids_list)} IDs but {len(scores_list)} scores")
        
if SUBMIT_TO_SCORE_BOARD:

    load_dotenv()
    TOKEN = os.getenv("TOKEN")

    if os.path.exists("./scores.csv"):
        response = requests.post("http://34.122.51.94:9090/mia", files={"file": open("scores.csv", "rb")}, headers={"token": TOKEN})
        print(response.json())
    else:
        print("scores.csv file does not exist. Make sure to run inference phase with rmia_offline by setting INFERENCE equal to True.")



