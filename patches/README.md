# Third-party setup (not vendored — reproduce locally)

`third_party/` is git-ignored. Recreate it like this.

## Uni3D (Stage B classifier)

```bash
mkdir -p third_party && cd third_party
git clone https://github.com/baaivision/Uni3D.git
cd Uni3D
git checkout 64e03c3c42c196e8cb5ed03857810af9fc9ac39c
# Apply our patch: replaces pointnet2_ops (CUDA) FPS with a pure-PyTorch one,
# so no nvcc / extension build is needed (works on Blackwell sm_120).
git apply ../../patches/uni3d_point_encoder_no_cuda.patch
```

### Weights (git-ignored, see ../weights/)
- `weights/uni3d-b.pt`  — Uni3D-B point encoder checkpoint (BAAI release)
- `weights/eva02_e_clip.bin` — EVA02-E-14-plus open_clip text/vision weights
