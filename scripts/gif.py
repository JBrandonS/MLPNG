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
    - Transparent background for clean compositing.
"""

import math

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# ── Settings ──────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 1280
FRAMES = 160
DURATION = 40  # ms per frame (~25 fps)
FNL = 200.0  # fixed local f_NL throughout animation
R_START = 5  # initial sphere radius (px)
R_END = 500  # final sphere radius (px)
TEX_SIZE = 1024  # resolution of the pre-generated texture map
CX, CY = WIDTH // 2, HEIGHT // 2

# ── CMB-like multi-scale Gaussian random field ────────────────────────────────
# Approximate Sachs-Wolfe CMB angular power spectrum with several Gaussian
# smoothing scales weighted by a rough C_ℓ ∝ 1/ℓ(ℓ+1) envelope.
rng = np.random.default_rng(42)
field = np.zeros((TEX_SIZE, TEX_SIZE))
for sigma, amp in [
    (2, 0.35),
    (5, 0.85),
    (12, 1.30),
    (25, 1.20),
    (50, 0.80),
    (90, 0.40),
]:
    field += amp * gaussian_filter(
        rng.standard_normal((TEX_SIZE, TEX_SIZE)), sigma=sigma
    )
field /= field.std()  # unit-variance Gaussian field


# ── Planck-inspired CMB diverging colormap ────────────────────────────────────
# cold (deep blue) → blue → white → orange-red → dark red (hot)
def make_cmb_cmap(n: int = 2048) -> np.ndarray:
    c = np.zeros((n, 3))
    for cmap_i in range(n):
        frac = cmap_i / (n - 1)
        if frac < 0.25:  # deep-blue → blue
            s = frac / 0.25
            c[cmap_i] = [0.04 + 0.06 * s, 0.04 + 0.10 * s, 0.55 + 0.45 * s]
        elif frac < 0.50:  # blue → white
            s = (frac - 0.25) / 0.25
            c[cmap_i] = [s, s, 1.0]
        elif frac < 0.75:  # white → red
            s = (frac - 0.50) / 0.25
            c[cmap_i] = [1.0, 1.0 - s, 1.0 - s]
        else:  # red → dark red
            s = (frac - 0.75) / 0.25
            c[cmap_i] = [1.0 - 0.20 * s, 0.04 * s, 0.04 * s]
    return (np.clip(c, 0, 1) * 255).astype(np.uint8)


CMAP = make_cmb_cmap()


def sample_texture_on_sphere(
    tex: np.ndarray,
    nx_map: np.ndarray,
    ny_map: np.ndarray,
    nz_map: np.ndarray,
    visible_mask: np.ndarray,
    rot_angle: float,
) -> np.ndarray:
    """Sample the texture with spherical coordinates and bilinear filtering."""
    cos_r = math.cos(rot_angle)
    sin_r = math.sin(rot_angle)

    # Inverse-rotate normals into texture space (stable surface rotation).
    x_tex = nx_map[visible_mask] * cos_r + nz_map[visible_mask] * sin_r
    y_tex = ny_map[visible_mask]
    z_tex = -nx_map[visible_mask] * sin_r + nz_map[visible_mask] * cos_r

    lon = np.arctan2(x_tex, z_tex)
    lat = np.arcsin(np.clip(y_tex, -1.0, 1.0))

    tex_h, tex_w = tex.shape
    u = (lon / (2.0 * math.pi) + 0.5) * tex_w
    v = (0.5 - lat / math.pi) * (tex_h - 1)

    u_floor = np.floor(u)
    v_floor = np.floor(v)
    u0 = u_floor.astype(np.int32) % tex_w
    v0 = np.clip(v_floor.astype(np.int32), 0, tex_h - 1)
    u1 = (u0 + 1) % tex_w
    v1 = np.clip(v0 + 1, 0, tex_h - 1)

    fu = u - u_floor
    fv = v - v_floor

    top = (1.0 - fu) * tex[v0, u0] + fu * tex[v0, u1]
    bottom = (1.0 - fu) * tex[v1, u0] + fu * tex[v1, u1]
    samples = (1.0 - fv) * top + fv * bottom

    sampled = np.zeros_like(nx_map, dtype=np.float32)
    sampled[visible_mask] = samples.astype(np.float32)
    return sampled


def rgba_to_gif_frame(frame: Image.Image) -> Image.Image:
    """Convert an RGBA frame to paletted GIF with index 0 as transparent."""
    rgba = np.array(frame, dtype=np.uint8)
    alpha = rgba[..., 3]

    rgb = Image.fromarray(rgba[..., :3], mode="RGB")
    quantize_enum = getattr(Image, "Quantize", None)
    quantize_method = getattr(quantize_enum, "MEDIANCUT", 0)
    quantized = rgb.quantize(colors=255, method=quantize_method)

    q_idx = np.array(quantized, dtype=np.uint8) + 1
    q_idx[alpha == 0] = 0

    gif_frame = Image.fromarray(q_idx, mode="P")
    base_palette = quantized.getpalette() or [0] * 768
    base_palette = base_palette[: 255 * 3]
    palette = [0, 0, 0] + base_palette
    palette.extend([0] * (768 - len(palette)))
    gif_frame.putpalette(palette[:768])
    return gif_frame


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

    # Spherical remap + bilinear filtering avoids apparent stretch during spin.
    g = sample_texture_on_sphere(field, nx, ny, nz, mask, rot)

    # Sample Gaussian field, then apply local PNG model
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

    # ── Composite sphere over fully transparent background ────────────────────
    final_arr = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    final_arr[mask, :3] = sphere[mask]
    final_arr[mask, 3] = 255
    final = Image.fromarray(final_arr, mode="RGBA")

    images.append(final)

    if (i + 1) % 40 == 0:
        print(f"  Frame {i+1}/{FRAMES}  R={R:.0f}")

# ── Save GIF ──────────────────────────────────────────────────────────────────
gif_frames = [rgba_to_gif_frame(frame) for frame in images]

gif_frames[0].save(
    "inflating_sphere.gif",
    save_all=True,
    append_images=gif_frames[1:],
    duration=DURATION,
    loop=0,
    optimize=True,
    transparency=0,
    disposal=2,
)
print("Saved inflating_sphere.gif")
