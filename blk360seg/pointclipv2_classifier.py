"""PointCLIP V2 open-vocabulary classifier (depth-projection baseline).

Renders each object into 10 smoothed depth views (Realistic Projection), encodes
them with a CLIP *image* encoder, averages the view features, and matches against
class text embeddings. Mirrors Uni3DClassifier's interface so the two can be
compared on identical objects. By default it reuses the SAME EVA02-E CLIP +
templates as Uni3D, so the only difference is native-3D (Uni3D) vs depth-projection.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .realistic_projection import RealisticProjection


class PointCLIPv2Classifier:
    def __init__(self, clip_ckpt, device="cuda", npoints=10000,
                 clip_model="EVA02-E-14-plus"):
        import open_clip
        self.device = device
        self.npoints = npoints

        self.clip, _, preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_ckpt)
        self.clip = self.clip.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(clip_model)

        # image size + normalization pulled from the model's own preprocess
        self.image_size = self.clip.visual.image_size
        if isinstance(self.image_size, (tuple, list)):
            self.image_size = self.image_size[0]
        norm = [t for t in preprocess.transforms if t.__class__.__name__ == "Normalize"][0]
        self.mean = torch.tensor(norm.mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(norm.std, device=device).view(1, 3, 1, 1)

        self.proj = RealisticProjection(device=device)
        self.text_embed = None
        self.class_names = None

    @torch.no_grad()
    def set_classes(self, class_names, templates=(
            "a point cloud of a {}.", "a point cloud of the {}.",
            "a 3d model of a {}.", "a 3d model of the {}.",
            "a 3d point cloud model of a {}.", "a photo of a {}.",
            "a {}.", "the {}.", "a {} object.", "there is a {} in the scene.")):
        embs = []
        for c in class_names:
            toks = self.tokenizer([t.format(c) for t in templates]).to(self.device)
            te = self.clip.encode_text(toks).float()
            te = te / te.norm(dim=-1, keepdim=True)
            embs.append(te.mean(0))
        t = torch.stack(embs)
        self.text_embed = (t / t.norm(dim=-1, keepdim=True)).to(self.device)
        self.class_names = list(class_names)

    def _prep(self, xyz):
        n = len(xyz)
        idx = np.random.choice(n, self.npoints, replace=(n < self.npoints))
        xyz = xyz[idx].astype(np.float32)
        xyz = xyz - xyz.mean(0)
        xyz = xyz / (np.max(np.linalg.norm(xyz, axis=1)) + 1e-9)
        return torch.from_numpy(xyz).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _encode_pc(self, xyz):
        pts = self._prep(xyz)                       # [1, N, 3]
        img = self.proj.get_img(pts)                # [V, 3, 112, 112], in [0,1]
        img = F.interpolate(img, size=self.image_size, mode="bilinear",
                            align_corners=False)
        img = (img - self.mean) / self.std
        feat = self.clip.encode_image(img).float()  # [V, D]
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feat = feat.mean(0, keepdim=True)           # fuse views
        return feat / feat.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def classify(self, xyz, rgb=None, topk=3):
        assert self.text_embed is not None, "call set_classes() first"
        pe = self._encode_pc(xyz)
        scale = self.clip.logit_scale.exp().float()
        probs = (scale * (pe @ self.text_embed.T)).squeeze(0).softmax(-1)
        vals, idx = probs.topk(min(topk, len(self.class_names)))
        return [(self.class_names[int(i)], float(v)) for v, i in zip(vals, idx)]
