import dgl
import torch
from torch.utils.data import Dataset, DataLoader


class DGLGraphDataset(Dataset):

    def __init__(self, graphs, global_feats):
        assert isinstance(graphs, list), "expected spec of 'graphs' is a list of graphs"

        self.graphs = graphs
        self.global_feats = global_feats
        self.len = len(graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.global_feats[idx]

    def __len__(self):
        return self.len


class DGLGraphDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super(DGLGraphDataLoader, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        batched_gs = dgl.batch([item[0] for item in batch])
        batched_global_feats = torch.stack([item[1] for item in batch])
        return batched_gs, batched_global_feats
