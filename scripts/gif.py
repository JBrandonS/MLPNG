"""
Inflating sphere GIF visualising primordial non-Gaussianity growth.

Physics:
  - Sphere inflates from R_START → R_END (quantum fluctuations stretched to
    superhorizon scales).
  - Temperature field starts purely Gaussian (fnl = 0) and evolves via the
    local PNG model:  Φ = φ_G + (fnl/100)(φ_G² − ⟨φ_G²⟩)
    producing asymmetrically bright hot spots as fnl grows.
  - Positive fnl → hot spots amplified quadratically, cold spots suppressed.

Physics (local PNG model):
  Φ(x) = φ_G(x) + (f_NL/100) * [φ_G(x)² − ⟨φ_G²⟩]

  φ_G is a unit-variance Gaussian random field approximating the primordial
  curvature power spectrum.  As inflation proceeds f_NL grows 0 → FNL_MAX,
  quadratically amplifying hot spots while suppressing cold spots — the
  characteristic positive skewness of local-type PNG.

  Caveats:
  - Single-field slow-roll inflation predicts |f_NL| ≈ O(ε) ≪ 1 (Maldacena 2003).
    Planck 2018 constrains f_NL^local = −0.9 ± 5.1.  The large f_NL here is
    illustrative, showing the qualitative effect at visible amplitude.
  - ΔT/T ≈ Φ/5 (Sachs-Wolfe), so colour ∝ Φ.
  - Zero mean is preserved via the variance subtraction ⟨φ_G²⟩.

Rendering:
  - Orthographic projection: sphere face maps to a single sky patch.
  - Lambert diffuse + limb darkening → 3-D look (no specular highlight).
  - Planck-inspired CMB colormap (blue → white → red).
  - Very slow rotation (~25° total) to reinforce depth.
  - Atmospheric glow; plain dark background.
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter

np.random.seed(42)

# ── Settings ──────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 640, 640
FRAMES = 220
DURATION = 40  # ms per frame (~25 fps)
FNL = 200.0  # fixed local f_NL throughout animation
R_START = 55  # initial sphere radius (px)
R_END = 220  # final sphere radius (px)
TEX_SIZE = 512  # resolution of the pre-generated texture map
CX, CY = WIDTH // 2, HEIGHT // 2

# ── CMB-like multi-scale Gaussian random field ────────────────────────────────
# Approximate Sachs-Wolfe CMB angular power spectrum with several Gaussian
# smoothing scales weighted by a rough C_ℓ ∝ 1/ℓ(ℓ+1) envelope.
rng = np.random.RandomState(42)
field = np.zeros((TEX_SIZE, TEX_SIZE))
for sigma, amp in [
    (2, 0.35),
    (5, 0.85),
    (12, 1.30),
    (25, 1.20),
    (50, 0.80),
    (90, 0.40),
]:
    field += amp * gaussian_filter(rng.randn(TEX_SIZE, TEX_SIZE), sigma=sigma)
field /= field.std()  # unit-variance Gaussian field


# ── Planck-inspired CMB diverging colormap ────────────────────────────────────
# cold (deep blue) → blue → white → orange-red → dark red (hot)
def make_cmb_cmap(n: int = 2048) -> np.ndarray:
    c = np.zeros((n, 3))
    for i in range(n):
        t = i / (n - 1)
        if t < 0.25:  # deep-blue → blue
            s = t / 0.25
            c[i] = [0.04 + 0.06 * s, 0.04 + 0.10 * s, 0.55 + 0.45 * s]
        elif t < 0.50:  # blue → white
            s = (t - 0.25) / 0.25
            c[i] = [s, s, 1.0]
        elif t < 0.75:  # white → red
            s = (t - 0.50) / 0.25
            c[i] = [1.0, 1.0 - s, 1.0 - s]
        else:  # red → dark red
            s = (t - 0.75) / 0.25
            c[i] = [1.0 - 0.20 * s, 0.04 * s, 0.04 * s]
    return (np.clip(c, 0, 1) * 255).astype(np.uint8)


CMAP = make_cmb_cmap()

# ── Orthographic patch scale (fraction of texture shown across sphere face) ──
PATCH_SCALE = 0.42  # ~42 % of TEX_SIZE → one coherent sky patch

# ── Precomputed pixel grid ────────────────────────────────────────────────────
Yg, Xg = np.mgrid[0:HEIGHT, 0:WIDTH]
DX = (Xg - CX).astype(np.float32)
DY = (Yg - CY).astype(np.float32)

# ── Lighting (fixed direction: upper-left, slightly toward viewer) ────────────
light = np.array([0.45, -0.40, 0.80], dtype=np.float32)
light /= np.linalg.norm(light)

# ── Main render loop ──────────────────────────────────────────────────────────
images = []

for i in range(FRAMES):
    t = i / (FRAMES - 1)
    te = t * t * (3 - 2 * t)  # smoothstep: slow start and end

    R = float(R_START + (R_END - R_START) * te)
    rot = 0.44 * te  # ~25° total — slow drift to show depth

    r2 = DX**2 + DY**2
    mask = r2 <= R * R

    # Surface normals (outward unit vectors on sphere surface)
    nx = np.where(mask, DX / R, 0.0).astype(np.float32)
    ny = np.where(mask, DY / R, 0.0).astype(np.float32)
    nz = np.where(
        mask, np.sqrt(np.clip(1.0 - nx**2 - ny**2, 0.0, 1.0)), 0.0
    ).astype(np.float32)

    # Orthographic single-patch projection with slow y-axis rotation.
    # Rotate (nx, nz) by `rot`, then map (rot_nx, ny) directly onto a central
    # patch of the texture — no wrapping around the back, one coherent region.
    cos_r = math.cos(rot)
    sin_r = math.sin(rot)
    rot_nx = nx * cos_r - nz * sin_r  # rotated x component
    # ny is unchanged (rotation is around the vertical/y axis)

    u = np.clip(
        ((rot_nx * PATCH_SCALE + 0.5) * (TEX_SIZE - 1)).astype(np.int32),
        0,
        TEX_SIZE - 1,
    )
    v = np.clip(
        ((ny * PATCH_SCALE + 0.5) * (TEX_SIZE - 1)).astype(np.int32), 0, TEX_SIZE - 1
    )

    # Sample Gaussian field, then apply local PNG model
    g = field[v, u]  # unit-variance Gaussian
    phi = g + (FNL / 100.0) * (g**2 - 1.0)  # Φ = φ_G + fnl*(φ_G²−1)

    # Fixed colour stretch
    vmin = -2.8
    vmax = 4.3  # asymmetric: hot spots extend further with positive f_NL
    norm = np.clip((phi - vmin) / (vmax - vmin), 0.0, 1.0)

    idx = np.clip((norm * (len(CMAP) - 1)).astype(np.int32), 0, len(CMAP) - 1)
    color = CMAP[idx]  # (H, W, 3) uint8

    # ── 3-D shading (diffuse + limb darkening, no specular) ───────────────────
    diffuse = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0.0, 1.0)
    limb = np.clip(nz, 0.0, 1.0) ** 0.40  # limb darkening at edges

    shading = (0.10 + 0.52 * diffuse + 0.38 * limb)[..., np.newaxis]
    sphere = np.clip(color.astype(np.float32) * shading, 0, 255).astype(np.uint8)

    # ── Background: plain dark ────────────────────────────────────────────────
    bg = np.full((HEIGHT, WIDTH, 3), (4, 4, 14), dtype=np.uint8)

    # ── Atmospheric glow (blurred ring just outside sphere) ───────────────────
    glow_r = int(R + 22)
    glow_img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    ImageDraw.Draw(glow_img).ellipse(
        [CX - glow_r, CY - glow_r, CX + glow_r, CY + glow_r],
        fill=(18, 8, 55),
    )
    glow_arr = np.array(
        glow_img.filter(ImageFilter.GaussianBlur(radius=int(18 + 8 * te)))
    )
    outside = ~mask
    bg[outside] = np.clip(
        bg[outside].astype(np.int16) + glow_arr[outside], 0, 255
    ).astype(np.uint8)

    # ── Composite sphere over background ──────────────────────────────────────
    mask3 = mask[..., np.newaxis]
    final_arr = np.where(mask3, sphere, bg)
    final = Image.fromarray(final_arr.astype(np.uint8), mode="RGB")

    # ── HUD ───────────────────────────────────────────────────────────────────
    draw = ImageDraw.Draw(final)
    # draw.text((14, 14), f"f_NL = {FNL:+.0f}", fill=(230, 230, 230))
    # draw.text((14, 30), f"Inflation: {te * 100:.0f}%", fill=(160, 160, 160))

    images.append(final)

    if (i + 1) % 40 == 0:
        print(f"  Frame {i+1}/{FRAMES}  R={R:.0f}")

# ── Save GIF ──────────────────────────────────────────────────────────────────
images[0].save(
    "inflating_sphere.gif",
    save_all=True,
    append_images=images[1:],
    duration=DURATION,
    loop=0,
    optimize=False,
)
print("Saved inflating_sphere.gif")
