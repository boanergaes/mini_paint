# Mini Paint

Interactive 2D vector drawing application built with Python and OpenGL.

## Requirements

- Python 3.10+
- OpenGL 3.3 capable graphics drivers

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Features

- **Vector shapes**: Lines, polylines, and regular polygons are stored as mathematical objects (vertices, color, transform matrix) and redrawn with OpenGL each frame.
- **Coordinate mapping**: Mouse screen coordinates are converted to world coordinates through a custom viewport and orthographic projection matrix.
- **Affine transforms**: Select a shape and apply translation, rotation, or scaling in real time using 2D affine matrices.
- **HUD menu**: On-screen text menu built programmatically (no external font assets).

## Controls

| Key | Action |
|-----|--------|
| `1` | Line tool |
| `2` | Polyline tool |
| `3` | Regular polygon tool |
| `4` | Select tool |
| `5` | Transform tool |
| `T` | Translate mode |
| `R` | Rotate mode |
| `S` | Scale mode |
| `C` | Cycle draw color |
| `[` / `]` | Decrease / increase polygon sides |
| `Enter` or right-click | Finish current polyline |
| `Delete` / `Backspace` | Delete selected shape |
| `Esc` | Cancel current draft |

### Drawing

- **Line**: click start point, then end point.
- **Polyline**: click each vertex, then press `Enter` or right-click to finish.
- **Polygon**: click center, then click to set radius and orientation.
- **Select**: click near a shape to select it.
- **Transform**: select a shape, choose `T`/`R`/`S`, then drag with the left mouse button. Hold `Shift` while scaling for uniform scaling.
