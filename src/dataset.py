import os, json, torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CXRDataset(Dataset):
    def __init__(self, ann_json, img_root, tokenizer,
                 img_size=512, max_len=512, split='train', two_view=True):
        with open(ann_json, 'r', encoding='utf-8') as f:
            items = json.load(f)
        self.items = [x for x in items if x['split'] == split]
        self.img_root = img_root
        self.two_view = two_view
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def _load_img(self, rel):
        p = os.path.join(self.img_root, rel)
        return self.tf(Image.open(p).convert('RGB'))

    def __getitem__(self, idx):
        it = self.items[idx]
        imgs = it['images']
        img1 = self._load_img(imgs[0])
        if self.two_view and len(imgs) > 1:
            img2 = self._load_img(imgs[1])
        else:
            img2 = torch.zeros_like(img1)

        text = it['report']
        tok = self.tokenizer(text, max_length=self.max_len, truncation=True,
                             padding="max_length", return_tensors="pt")
        labels = tok["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "img1": img1, "img2": img2,
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "labels": labels
        }

    def __len__(self):
        return len(self.items)
