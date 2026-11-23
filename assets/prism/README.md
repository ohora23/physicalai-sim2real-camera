# Prism Target

## Build parameters

| Parameter | Value |
|-----------|-------|
| Faces | 6 |
| Board size | 5 x 5 squares |
| Square size | 6.0 cm |
| Marker size | 4.5 cm |
| Working distance | 0.75 m |
| Focal length | 600 px |
| Apothem | 0.3118 m |
| Face width | 36.0 cm |
| Face height | 36.0 cm |

## Formulas

- `square_m = max(0.05, cell_px * 6 cells / 0.75 * distance_m / focal_px)` rounded up to 5 mm (cell_px 6 by default)
- `marker_m = 0.75 * square_m`
- `w = cols * square_m + 2 * margin` (margin = 0.5 * square_m)
- `apothem = w / (2 * tan(pi / N))`
- Face k angle: `phi_k = 2*pi*k/N`

## Files

- `prism.json` — target configuration (boards, transforms, wedge)
- `prism.xml` — MuJoCo model with textures and geoms
- `face_<k>.png` — ChArUco board texture for face k
- `README.md` — this file

## Build notes

- Print at 1:1 scale; measure edge lengths to verify.
- Use a rigid board (not paper) to avoid warping.
- Laser-cut or 3D-print the prism frame from the apothem and height.
