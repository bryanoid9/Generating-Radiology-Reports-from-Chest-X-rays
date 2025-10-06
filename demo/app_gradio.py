import os, torch, gradio as gr
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from src.models.baseline_resnet_gpt2 import ReportGen  # if run from project root

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

# Path to your trained checkpoint
CKPT = "runs/baseline/best.pt"
model = ReportGen().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

tf = transforms.Compose([
    transforms.Resize(512), transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

RED_FLAGS = ["pneumothorax", "aortic dissection", "massive", "widened mediastinum", "malpositioned"]

def infer(img1, img2):
    if img1 is None:
        return "Upload at least one image."
    t1 = tf(img1.convert("RGB")).unsqueeze(0).to(device)
    t2 = tf(img2.convert("RGB")).unsqueeze(0).to(device) if img2 else None
    text = model.generate(tokenizer, t1, t2, max_new_tokens=220)
    banner = "AI-generated draft for radiologist review — NOT for clinical use."
    if any(k in text.lower() for k in RED_FLAGS):
        banner = "⚠️ Urgent human review suggested. " + banner
    return banner + "\n\n" + text

demo = gr.Interface(
    fn=infer,
    inputs=[gr.Image(type="pil", label="PA / AP view"),
            gr.Image(type="pil", label="Lateral view (optional)")],
    outputs=gr.Textbox(lines=20, label="Draft Report"),
    title="Chest X-ray → Report (Research Prototype)"
)

if __name__ == "__main__":
    demo.launch()

