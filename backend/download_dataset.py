"""
TruthLens - Dataset Download Script
Downloads the Kaggle Fake News Dataset.

Usage:
    python download_dataset.py

The dataset consists of two CSV files:
    - True.csv  (Real news articles)
    - Fake.csv  (Fake news articles)

Source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
"""

import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def main():
    print("=" * 60)
    print("  TruthLens — Dataset Download")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    fake_path = os.path.join(DATA_DIR, 'Fake.csv')
    true_path = os.path.join(DATA_DIR, 'True.csv')

    if os.path.exists(fake_path) and os.path.exists(true_path):
        print("  Dataset already downloaded!")
        print(f"  Location: {DATA_DIR}")
        return

    print()
    print("  Please download the dataset manually from Kaggle:")
    print()
    print("  1. Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    print("  2. Click 'Download' (requires Kaggle account)")
    print("  3. Extract the ZIP file")
    print(f"  4. Place 'True.csv' and 'Fake.csv' in: {DATA_DIR}")
    print()
    print("  Alternatively, use the Kaggle CLI:")
    print("    pip install kaggle")
    print("    kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset")
    print(f"    # Extract to {DATA_DIR}")
    print()

    # Try kaggle CLI if available
    try:
        import kaggle
        print("  Kaggle package detected! Attempting automatic download...")
        os.system(f'kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p "{DATA_DIR}" --unzip')

        if os.path.exists(fake_path) and os.path.exists(true_path):
            print("  Download successful!")
        else:
            print("  Automatic download may have failed. Please download manually.")
    except ImportError:
        print("  Install kaggle CLI for automatic download: pip install kaggle")

    print("=" * 60)


if __name__ == '__main__':
    main()
