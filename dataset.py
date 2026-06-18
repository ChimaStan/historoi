from torch.utils.data import Dataset

class WSIDataset(Dataset):
    def __init__(self, df, wsi, transform, level, ps, half_ps_level0):
        self.wsi = wsi
        self.transform = transform
        self.df = df
        self.level = level
        self.ps = ps
        self.half_ps_level0 = half_ps_level0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        center_x = int(self.df.iloc[idx, 0])
        center_y = int(self.df.iloc[idx, 1])

        x = center_x - self.half_ps_level0
        y = center_y - self.half_ps_level0

        x = max(x, 0)
        y = max(y, 0)
        
        patch = self.wsi.read_region((x, y), self.level, (self.ps, self.ps))
        patch = self.transform(patch.convert('RGB'))

        return patch
