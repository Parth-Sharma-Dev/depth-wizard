from huggingface_hub import list_repo_files


REPO_ID = "earthflow/GAMUS"
SCENE = "DC_03_26"


def main():

    print("=" * 70)
    print("GAMUS Scene Inspection")
    print("=" * 70)

    files = list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    matches = [
        f for f in files
        if SCENE in f
    ]

    print(f"\nFiles containing '{SCENE}':")
    print("-" * 70)

    for file in matches:
        print(file)

    print(f"\nTotal matching files: {len(matches)}")


if __name__ == "__main__":
    main()