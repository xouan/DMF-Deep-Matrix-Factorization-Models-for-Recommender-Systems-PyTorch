import torch
from torch.utils.data import DataLoader, Dataset
class UserItemRatingDataset(Dataset):
    def __init__(self, user, item, target):
        self.user = torch.LongTensor(user)
        self.item = torch.LongTensor(item)
        self.target = torch.FloatTensor(target)
        
    def __getitem__(self, index):
        return self.user[index], self.item[index], self.target[index]
    
    def __len__(self):
        return self.user.size(0)