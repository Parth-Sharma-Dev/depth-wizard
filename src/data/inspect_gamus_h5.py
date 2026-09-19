from pathlib import Path

import h5py


DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "gamus_inspection"
)

FILES = [
    DATA_DIR / "images" / "test" / "DC_03_26_RGB.h5",
    DATA_DIR / "heights" / "test" / "DC_03_26_AGL.h5",
    DATA_DIR / "classes" / "test" / "DC_03_26_CLS.h5",
]


def inspect_dataset(name, obj):

    if isinstance(obj, h5py.Dataset):

        print(f"\nDataset: {name}")
        print(f"  Shape : {obj.shape}")
        print(f"  Dtype : {obj.dtype}")

        if obj.shape and all(dim > 0 for dim in obj.shape):
            print(f"  Chunks: {obj.chunks}")

        print(f"  Attrs : {dict(obj.attrs)}")


def inspect_file(path):

    print("\n" + "=" * 70)
    print(path.name)
    print("=" * 70)

    if not path.exists():
        print("FILE NOT FOUND")
        return

    with h5py.File(path, "r") as file:

        print("\nTop-level keys:")

        for key in file.keys():
            print(f"  {key}")

        file.visititems(inspect_dataset)


def main():

    for path in FILES:
        inspect_file(path)


if __name__ == "__main__":
    main()