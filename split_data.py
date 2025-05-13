from torch.utils.data import Subset

import numpy as np


def split_dataset_balanced(dataset, k):
    """
    Split a dataset into k disjoint subsets with equal numbers of members and non-members.
    The number of data points must be divisible by k.
    
    Args:
        dataset: PyTorch dataset (MembershipDataset) returning (id, img, label, membership)
        k: Number of subsets
    
    Returns:
        List of k Subset objects, each with balanced membership
    
    Raises:
        ValueError: If k is invalid or total_per_class is not divisible by k
    """
    if k <= 0 or k > len(dataset):
        raise ValueError("k must be positive and not exceed dataset size")
    
    indices = np.arange(len(dataset))
    memberships = np.array([dataset[i][3] for i in indices])  
    
    member_indices = indices[memberships == 1]
    non_member_indices = indices[memberships == 0]
       
    total_per_class = len(member_indices)  
    
    if total_per_class % k != 0:
        raise ValueError(f"Number of members ({total_per_class}) must be divisible by k ({k})")
    
    np.random.shuffle(member_indices)
    np.random.shuffle(non_member_indices)
    
    per_subset_per_class = total_per_class // k
    
    subsets = []
    start_member = 0
    start_non_member = 0
    
    for _ in range(k):
        member_subset = member_indices[start_member:start_member + per_subset_per_class]
        non_member_subset = non_member_indices[start_non_member:start_non_member + per_subset_per_class]
        
        subset_indices = np.concatenate([member_subset, non_member_subset])
        np.random.shuffle(subset_indices)  
        
        subsets.append(Subset(dataset, subset_indices))
        
        start_member += per_subset_per_class
        start_non_member += per_subset_per_class
    
    return subsets