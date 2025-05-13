import numpy as np
import math 
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

def sm_taylor_softmax(logits, label, m=0.6, n=4, T=2.0):
    logits = logits/T
    c_y = logits[label]
    apx_c_y = sum([(c_y - m) ** i / math.factorial(i) for i in range(n + 1)])
    apx_others = sum([sum([c_i ** i / math.factorial(i) for i in range(n + 1)]) for c_i in logits])

    confidence = apx_c_y / (apx_c_y  + apx_others)
    return confidence.clamp(0,1)

def compute_pr_x_theta(model, dataloader):
    print("Starting compute_pr_x_theta...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    pr_x_theta = []
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Processing batches"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            for i in range(images.size(0)):
                confidence = sm_taylor_softmax(outputs[i], labels[i], m=0.6, n=4, T=2.0)
                pr_x_theta.append(confidence.item())
    
    print("compute_pr_x_theta completed.")
    return np.array(pr_x_theta)

def rmia_offline(target_model, shadow_models, priv_dataloader, pub_dataloader, 
                 gamma=2.0, a=0.3, num_z_samples=2500):
    print(f"Starting rmia_offline with gamma={gamma}, a={a}, num_z_samples={num_z_samples}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_model.eval()
    for model in shadow_models:
        model.eval()

    public_data = [(id, img, label, membership) 
                   for id, img, label, membership in pub_dataloader.dataset if membership == 0]
    
    if len(public_data) > num_z_samples:
        public_data = np.random.choice(public_data, size=num_z_samples, replace=False)

    z_images = torch.stack([item[1] for item in public_data]).to(device)
    z_labels = torch.tensor([item[2] for item in public_data]).to(device)
    z_loader = DataLoader(list(zip(z_images, z_labels, [0]*len(z_images))),
                          batch_size=64, shuffle=False)
    
    pr_z_theta_target = compute_pr_x_theta(target_model, z_loader)
    pr_z_ref = np.mean([compute_pr_x_theta(model, z_loader) for model in tqdm(shadow_models, desc="Processing shadow models")], axis=0)

    scores = []

    with torch.no_grad():
        for ids, images, labels, _ in tqdm(priv_dataloader, desc="Processing private data"):
            images, labels = images.to(device), labels.to(device)
            outputs = target_model(images)
            for i in range(images.size(0)):
                pr_x_theta = sm_taylor_softmax(outputs[i], labels[i], m=0.6, n=4, T=2.0).item()
                pr_x_out = np.mean([sm_taylor_softmax(model(images[i].unsqueeze(0))[0], labels[i], m=0.6, n=4, T=2.0).item() for model in shadow_models])
                pr_x = 0.5 * ((1 + a) * pr_x_out + (1 - a))

                ratio_x = pr_x_theta / pr_x
                ratio_z = pr_z_theta_target / pr_z_ref
                likelihood_ratio = ratio_x / ratio_z

                score = np.mean(likelihood_ratio >= gamma)
                scores.append((ids[i].item(), score))

    print(f"rmia_offline completed. Computed {len(scores)} scores.")
    return np.array(scores)

