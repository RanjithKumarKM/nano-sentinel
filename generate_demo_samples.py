"""
Nano Sentinel — synthetic wafer-map demo sample generator

Run from your project root:
    python generate_demo_samples.py --output backend/Incoming --count 100

Pass your actual monitored folder with --output if it differs.

IMPORTANT: These are synthetic demo inputs, not genuine WM-811K samples and
not valid evaluation data for claiming model accuracy. Confirm that your current
backend accepts 2D .npy maps encoded as 0=background, 1=good die, 2=defective die.
"""

from pathlib import Path
import argparse
import math
import numpy as np

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Near-full", "Random", "Scratch", "none"]
SIZE = 96


def base(size=SIZE):
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    radius = size * 0.44
    rr = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    wafer = rr <= radius
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[wafer] = 1
    return arr, wafer, rr, xx, yy, c, radius


def generate_pattern(label, rng, size=SIZE):
    arr, wafer, rr, xx, yy, c, radius = base(size)
    defect = np.zeros((size, size), dtype=bool)

    if label == "Center":
        rad = rng.uniform(size * .09, size * .17)
        defect = (xx-c)**2 + (yy-c)**2 <= rad**2
    elif label == "Donut":
        r0 = rng.uniform(size * .20, size * .27)
        thickness = rng.uniform(size * .035, size * .065)
        defect = np.abs(rr-r0) <= thickness
    elif label == "Edge-Loc":
        for _ in range(rng.integers(3, 7)):
            a = rng.uniform(0, 2*math.pi)
            r = radius * rng.uniform(.78, .98)
            cx, cy = c + r*math.cos(a), c + r*math.sin(a)
            cr = rng.uniform(size*.035, size*.075)
            defect |= (xx-cx)**2 + (yy-cy)**2 <= cr**2
    elif label == "Edge-Ring":
        r0 = radius * rng.uniform(.83, .94)
        thickness = rng.uniform(size*.035, size*.065)
        defect = np.abs(rr-r0) <= thickness
    elif label == "Loc":
        cx, cy = c+rng.uniform(-size*.20, size*.20), c+rng.uniform(-size*.20, size*.20)
        rad = rng.uniform(size*.045, size*.10)
        defect = (xx-cx)**2 + (yy-cy)**2 <= rad**2
    elif label == "Near-full":
        defect = wafer & (rng.random((size, size)) < rng.uniform(.62, .86))
    elif label == "Random":
        defect = wafer & (rng.random((size, size)) < rng.uniform(.025, .09))
    elif label == "Scratch":
        angle = rng.uniform(-.65, .65)
        offset = rng.uniform(-size*.16, size*.16)
        distance = np.abs((yy-c)*math.cos(angle) - (xx-c)*math.sin(angle) - offset)
        defect = distance <= rng.uniform(1.0, 2.2)
    elif label == "none":
        defect = np.zeros((size, size), dtype=bool)

    arr[defect & wafer] = 2
    if label not in ("none", "Near-full"):
        noise = wafer & (rng.random((size, size)) < .0025)
        arr[noise] = 2
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="backend/Incoming",
                        help="Folder monitored by your app (default: backend/Incoming)")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(args.count):
        label = CLASSES[i % len(CLASSES)]
        sample = generate_pattern(label, rng)
        # Filename intentionally does not reveal the intended synthetic pattern.
        np.save(out / f"demo_wafer_{i+1:04d}.npy", sample)

    print(f"Created {args.count} synthetic .npy demo files in: {out}")
    print("These are synthetic demonstration inputs, NOT genuine WM-811K test samples.")
    print("Verify that your current backend accepts this .npy format before presenting.")


if __name__ == "__main__":
    main()
