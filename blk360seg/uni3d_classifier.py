"""Uni3D open-vocabulary classifier (Stage B).

Loads the Uni3D point encoder + the EVA02-E CLIP text encoder, embeds an object
point cloud into CLIP space, and matches it against text prompts (class names).

Args match third_party/Uni3D/scripts/inference.sh for the uni3d-b checkpoint
(npoints 10000, group_size 64, num_group 512, pc_encoder_dim 512, embed_dim 1024,
pc_model eva02_base_patch14_448, pc_feat_dim 768).
"""
import os
import sys

import numpy as np
import torch

UNI3D_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "Uni3D")

UNI3D_B_ARGS = dict(
    pc_model="eva02_base_patch14_448", pretrained_pc=None, drop_path_rate=0.0,
    pc_feat_dim=768, embed_dim=1024, group_size=64, num_group=512,
    pc_encoder_dim=512, patch_dropout=0.0,
)


class Uni3DClassifier:
    def __init__(self, uni3d_ckpt, clip_ckpt, device="cuda", npoints=10000,
                 clip_model="EVA02-E-14-plus"):
        sys.path.insert(0, os.path.abspath(UNI3D_DIR))
        import open_clip
        from easydict import EasyDict
        from models import uni3d as uni3d_models

        self.device = device
        self.npoints = npoints

        model = uni3d_models.create_uni3d(EasyDict(UNI3D_B_ARGS))
        sd = torch.load(uni3d_ckpt, map_location="cpu", weights_only=False)
        for key in ("module", "model", "state_dict"):
            if isinstance(sd, dict) and key in sd:
                sd = sd[key]
                break
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[uni3d] loaded ckpt (missing={len(missing)}, unexpected={len(unexpected)})")
        self.model = model.to(device).eval()

        # CLIP (9.4 GB EVA02-E) is only needed to embed the class-name texts.
        # Load it lazily and cache embeddings on disk so repeat runs with a
        # known vocabulary never pay the ~25 GB RAM load peak (OOM-killer bait).
        self.clip = None
        self.clip_model_name = clip_model
        self.clip_ckpt = clip_ckpt
        self.text_embed = None
        self.class_names = None

    def _load_clip(self):
        import open_clip
        print("[uni3d] loading EVA02-E CLIP text encoder (slow)...")
        # build without weights, drop the 4.4 B-param vision tower (text-only
        # use), then mmap-load just the text weights: peak RAM ~19 GB instead
        # of ~28 GB (open_clip's own pretrained= path OOM-kills on 32 GB hosts)
        model = open_clip.create_model(self.clip_model_name, pretrained=None)
        model.visual = None
        sd = torch.load(self.clip_ckpt, map_location="cpu", mmap=True,
                        weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith("visual.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("visual.")]
        print(f"[uni3d] clip text weights loaded "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        assert not missing, f"text-tower keys missing: {missing[:5]}"
        self.clip = model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_name)

    @torch.no_grad()
    def set_classes(self, class_names, templates=(
            "a point cloud of a {}.", "a point cloud of the {}.",
            "a 3d model of a {}.", "a 3d model of the {}.",
            "a 3d point cloud model of a {}.", "a photo of a {}.",
            "a {}.", "the {}.", "a {} object.", "there is a {} in the scene.")):
        import hashlib
        import json
        key = hashlib.sha1(json.dumps(
            [self.clip_model_name, list(templates), list(class_names)]
        ).encode()).hexdigest()[:16]
        cache_dir = os.path.join(os.path.dirname(self.clip_ckpt),
                                 "text_embed_cache")
        cache = os.path.join(cache_dir, f"{key}.pt")
        if os.path.exists(cache):
            self.text_embed = torch.load(cache, map_location=self.device)
            self.class_names = list(class_names)
            print(f"[uni3d] text embeddings from cache ({cache})")
            return
        if self.clip is None:
            self._load_clip()
        embs = []
        for c in class_names:
            toks = self.tokenizer([t.format(c) for t in templates]).to(self.device)
            te = self.clip.encode_text(toks).float()
            te = te / te.norm(dim=-1, keepdim=True)
            embs.append(te.mean(0))
        t = torch.stack(embs)
        self.text_embed = (t / t.norm(dim=-1, keepdim=True)).to(self.device)
        self.class_names = list(class_names)
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(self.text_embed.cpu(), cache)
        print(f"[uni3d] text embeddings cached -> {cache}")

    def _prep(self, xyz, rgb):
        n = len(xyz)
        idx = np.random.choice(n, self.npoints, replace=(n < self.npoints))
        xyz, rgb = xyz[idx].astype(np.float32), rgb[idx].astype(np.float32)
        xyz = xyz - xyz.mean(0)
        xyz = xyz / (np.max(np.linalg.norm(xyz, axis=1)) + 1e-9)   # unit sphere
        pc = np.concatenate([xyz, rgb], axis=1)
        return torch.from_numpy(pc).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def classify(self, xyz, rgb, topk=3):
        assert self.text_embed is not None, "call set_classes() first"
        pe = self.model.encode_pc(self._prep(xyz, rgb)).float()
        pe = pe / pe.norm(dim=-1, keepdim=True)
        scale = self.model.logit_scale.exp().float()   # temperature from the ckpt
        probs = (scale * (pe @ self.text_embed.T)).squeeze(0).softmax(-1)
        vals, idx = probs.topk(min(topk, len(self.class_names)))
        return [(self.class_names[int(i)], float(v)) for v, i in zip(vals, idx)]
