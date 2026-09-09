#!/usr/bin/env python3
"""
process_dataset.py
Simple placeholder to detect the video file and show where to add processing.
Drop `singapore_mall.mp4` into this folder and run this script.
"""
import os
import sys


def main():
    fname = "singapore_mall.mp4"
    if not os.path.exists(fname):
        print(f"Place your video file named '{fname}' in this folder and rerun.")
        return 1

    print(f"Found '{fname}' — ready to implement processing pipeline.")
    # TODO: implement frame extraction, preprocessing, annotation, etc.
    return 0


if __name__ == "__main__":
    sys.exit(main())
