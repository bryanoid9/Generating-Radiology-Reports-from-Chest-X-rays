import torch, torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from transformers import GPT2LMHeadModel

class ImageEncoder(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base.children())[:-2])  # up to C5
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(2048, out_dim)

    def forward(self, x):
        f = self.backbone(x)       # (B, 2048, H, W)
        f = self.pool(f).flatten(1)  # (B, 2048)
        return self.proj(f)        # (B, out_dim)

class VisionConditionedGPT2(nn.Module):
    """
    Simple prepend-conditioning:
    project image features to K learned 'conditioning tokens'
    and prepend them to GPT-2 token embeddings.
    """
    def __init__(self, txt_model_name="gpt2", img_dim=1024, cond_tokens=8):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(txt_model_name)
        if self.gpt2.config.pad_token_id is None:
            self.gpt2.resize_token_embeddings(self.gpt2.config.vocab_size + 1)
            self.gpt2.config.pad_token_id = self.gpt2.config.vocab_size - 1
        hidden = self.gpt2.config.n_embd
        self.proj = nn.Linear(img_dim, hidden * cond_tokens)
        self.cond_tokens = cond_tokens
        self.hidden = hidden

    def forward(self, img_memory, input_ids, attention_mask=None, labels=None):
        B = img_memory.size(0)
        cond = self.proj(img_memory).view(B, self.cond_tokens, self.hidden)  # (B, Tc, H)
        tok_embeds = self.gpt2.transformer.wte(input_ids)                    # (B, T, H)
        inputs = torch.cat([cond, tok_embeds], dim=1)                        # (B, Tc+T, H)

        am = None
        if attention_mask is not None:
            am = torch.cat([
                torch.ones(B, self.cond_tokens, device=inputs.device, dtype=attention_mask.dtype),
                attention_mask
            ], dim=1)

        if labels is not None:
            pad = torch.full((B, self.cond_tokens), -100, device=labels.device, dtype=labels.dtype)
            labels = torch.cat([pad, labels], dim=1)

        return self.gpt2(inputs_embeds=inputs, attention_mask=am, labels=labels)

class ReportGen(nn.Module):
    def __init__(self, txt_model_name="gpt2", img_dim=1024, cond_tokens=8):
        super().__init__()
        self.enc = ImageEncoder(out_dim=img_dim)
        self.dec = VisionConditionedGPT2(txt_model_name, img_dim, cond_tokens)
        self.fuse = nn.Sequential(
            nn.Linear(img_dim * 2, img_dim),
            nn.ReLU(),
            nn.Linear(img_dim, img_dim)
        )

    def forward(self, img1, img2, input_ids, attention_mask, labels=None):
        f1 = self.enc(img1)
        f2 = self.enc(img2)
        fused = self.fuse(torch.cat([f1, f2], dim=1))
        return self.dec(fused, input_ids, attention_mask, labels)

    @torch.no_grad()
    def generate(self, tokenizer, img1, img2=None, max_new_tokens=220, temperature=0.7, top_p=0.9):
        if img2 is None:
            img2 = torch.zeros_like(img1)
        device = next(self.parameters()).device

        f1 = self.enc(img1.to(device))
        f2 = self.enc(img2.to(device))
        fused = self.fuse(torch.cat([f1, f2], dim=1))

        # seed prompt
        prompt = "FINDINGS: "
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # prepend conditioning tokens
        cond = self.dec.proj(fused).view(1, self.dec.cond_tokens, self.dec.hidden)
        inputs = torch.cat([cond, self.dec.gpt2.transformer.wte(ids)], dim=1)
        generated = ids

        for _ in range(max_new_tokens):
            out = self.dec.gpt2(inputs_embeds=inputs)
            logits = out.logits[:, -1, :]
            # nucleus sampling
            probs = torch.softmax(logits / temperature, dim=-1)
            sorted_probs, sorted_idx = probs.sort(descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            cut = (cum > top_p).nonzero()
            if cut.numel() > 0:
                k = cut[0, 1].item() + 1
                idx = torch.multinomial(sorted_probs[:, :k], 1)
                next_token = sorted_idx.gather(1, idx)
            else:
                next_token = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_token], dim=1)
            inputs = torch.cat([inputs, self.dec.gpt2.transformer.wte(next_token)], dim=1)

            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break

        text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        if "IMPRESSION:" not in text:
            text += "\nIMPRESSION: "
        return text.strip()
