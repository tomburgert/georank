from torch.utils.data import Dataset, Subset


class IndexDataset(Dataset):
    def __init__(self, dataset, indices=None):
        self.indices = indices
        if indices is not None:
            self.dataset = Subset(dataset, indices)
        else:
            self.dataset = dataset

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        if self.indices is None:
            orig_idx = idx
        else:
            orig_idx = self.dataset.indices[idx]
        return data, target, orig_idx

    def __len__(self):
        return len(self.dataset)
