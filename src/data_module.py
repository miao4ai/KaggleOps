from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.transforms as T
from PIL import Image
import os

class AnimalDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['filename'])  # Adjust to match your csv
        image = Image.open(image_path).convert('RGB')
        label = int(row['identity'])
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloaders(cfg):
    df = pd.read_csv(cfg['data']['csv_path'])
    df = df[df['split'] == cfg['data']['split']].dropna(subset=['identity'])

    # Shuffle & split
    df = df.sample(frac=1, random_state=cfg['project']['seed'])
    train_df = df.iloc[:int(0.8 * len(df))]
    val_df = df.iloc[int(0.8 * len(df)):]

    transform = T.Compose([
        T.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        T.ToTensor()
    ])

    train_ds = AnimalDataset(train_df, cfg['data']['image_dir'], transform)
    val_ds = AnimalDataset(val_df, cfg['data']['image_dir'], transform)

    train_loader = DataLoader(train_ds, batch_size=cfg['data']['batch_size'], shuffle=True,
                              num_workers=cfg['data']['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=cfg['data']['batch_size'], shuffle=False,
                            num_workers=cfg['data']['num_workers'])

    return train_loader, val_loader