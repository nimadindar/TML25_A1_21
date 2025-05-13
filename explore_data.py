import torch
from load_datasets import MembershipDataset

private_data: MembershipDataset = torch.load("./priv_out.pt")
public_data: MembershipDataset = torch.load("./pub.pt")


print(f"The private dataset has {len(private_data)} data points.")
print(f"The public dataset has {len(public_data)} data points.")

labels = torch.tensor(private_data.labels)  
num_classes = len(torch.unique(labels))
print(f"The dataset has {num_classes} classes.")

num_IN = sum(1 for membership in public_data.membership if membership == 1)
num_OUT = len(public_data) - num_IN

print(f"Number of IN data points: {num_IN} & Number of OUT data points: {num_OUT}")




