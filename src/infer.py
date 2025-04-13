import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from src.model_module import AnimalClassifier
import yaml

def predict():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    model = AnimalClassifier(cfg)
    model.load_state_dict(torch.load("best_model.pth"))  # or mlflow model
    model.eval()

    test_df = pd.read_csv("./data/test_metadata.csv")
    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor()
    ])

    preds = []
    for idx, row in test_df.iterrows():
        image = Image.open(f"./data/test_images/{row['filename']}").convert("RGB")
        image = transform(image).unsqueeze(0)
        pred = model(image).argmax(1).item()
        preds.append(pred)

    submission = pd.DataFrame({
        "filename": test_df['filename'],
        "identity": preds
    })
    submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    predict()