import os, math, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from dataset import CXRDataset
from models.baseline_resnet_gpt2 import ReportGen

def train(ann_json="data/annotations_iuxray.json",
          img_root="data/images_iuxray",
          out_dir="runs/baseline",
          batch_size=8,
          epochs_stage1=10,
          epochs_stage2=6):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    train_ds = CXRDataset(ann_json, img_root, tokenizer, split="train")
    val_ds   = CXRDataset(ann_json, img_root, tokenizer, split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = ReportGen().to(device)
    model.dec.gpt2.resize_token_embeddings(len(tokenizer))

    # Stage 1: freeze encoder
    for p in model.enc.parameters():
        p.requires_grad = False

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    steps = len(train_loader) * epochs_stage1
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=100, num_training_steps=steps)

    def run_epoch(loader, train_mode=True):
        model.train(train_mode)
        total = 0.0
        for batch in loader:
            img1 = batch["img1"].to(device)
            img2 = batch["img2"].to(device)
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(img1, img2, ids, attn, labels)
            loss = out.loss
            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); optim.zero_grad(); sched.step()
            total += loss.item() * img1.size(0)
        return total / len(loader.dataset)

    best_val = float("inf")

    # Stage 1
    for e in range(epochs_stage1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        print(f"[Stage1] Epoch {e+1}/{epochs_stage1}  train {tr:.3f}  val {va:.3f}")
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

    # Stage 2: unfreeze top encoder block + projection
    for name, p in model.enc.backbone.named_parameters():
        if "layer4" in name:
            p.requires_grad = True
    for p in model.enc.proj.parameters():
        p.requires_grad = True

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=0.01)

    for e in range(epochs_stage2):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        print(f"[Stage2] Epoch {e+1}/{epochs_stage2}  train {tr:.3f}  val {va:.3f}")
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

    print(f"Done. Best val loss: {best_val:.3f}. Saved to {os.path.join(out_dir, 'best.pt')}")

if __name__ == "__main__":
    train()
