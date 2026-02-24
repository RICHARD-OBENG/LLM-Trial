# Loading neccessary libraries
import subprocess
import sys
import h5py
import pathlib
import argparse
import shlex

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "out"

def run(cmd: str):
    print(f"\n>>>> {cmd}")
    args = shlex.split(cmd)
    res = subprocess.run(args, cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--visualize", action="store_true", help="run visualization scripts and save PNGs to ./out")
    args =p.parse_args()

    OUT.mkdir(exist_ok=True)


    # 1.2 sanity check: Numpy tiny example
    run("python attn_numpy_demo.py")

    # 1.31.4 unit test
    run("python -m pytest -q tests/test_attn_math.py")
    run("python -m pytest -q tests/test_casual_mask.py")

    # Matrix math walkthrough for MHA
    run("python demo_sha_shapes.py")

    if args.visualize:
        run("python demo_visialize_multi_head.py")
        print(f"\nVisualization images saved to: {OUT}")

    print("\nAll Part 1 demos/tests completed. ✅")

if __name__ == "__main__":
    main()