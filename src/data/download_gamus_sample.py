from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "earthflow/GAMUS"

FILES = [
    "images/test/DC_03_26_RGB.h5",
    "heights/test/DC_03_26_AGL.h5",
    "classes/test/DC_03_26_CLS.h5",
]

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "gamus_inspection"
)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("Downloading one GAMUS inspection scene")
    print("=" * 70)

    for filename in FILES:

        print(f"\nDownloading:")
        print(filename)

        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=OUTPUT_DIR,
        )

        print(f"Saved to: {local_path}")

    print("\nInspection scene downloaded successfully.")


if __name__ == "__main__":
    main()