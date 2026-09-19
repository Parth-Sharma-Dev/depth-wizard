from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .gamus_quality import QualityPolicy, build_height_targets
except ImportError:
    from gamus_quality import QualityPolicy, build_height_targets


class GAMUSDataset(Dataset):
    """
    GAMUS RGB -> AGL dataset.

    Expected manifest columns:
        scene
        split
        flags
        negative_pct
        gt50_pct

    The manifest is produced by build_gamus_manifest.py.

    Each sample returns:
        image       : float32 tensor [3, crop_size, crop_size], range [0, 1]
        height      : float32 tensor [crop_size, crop_size]
        valid_mask  : bool tensor [crop_size, crop_size]
        weight      : float32 tensor [crop_size, crop_size]
        scene       : scene identifier

    Training mode:
      - random spatial crops
      - optional horizontal/vertical flips

    Validation mode:
      - deterministic grid crops
    """

    def __init__(
        self,
        manifest,
        data_root: str | Path,
        split: str = "train",
        crop_size: int = 512,
        samples_per_scene: int = 16,
        training: bool = True,
        augment: bool = True,
        quality_policy: Optional[QualityPolicy] = None,
        image_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        max_cached_scenes: int = 1,
        seed: int = 42,
    ):
        self.manifest = manifest.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.split = split
        self.crop_size = crop_size
        self.samples_per_scene = samples_per_scene
        self.training = training
        self.augment = augment and training
        self.quality_policy = quality_policy or QualityPolicy()
        self.image_transform = image_transform
        self.max_cached_scenes = max(1, int(max_cached_scenes))
        self.seed = seed

        if len(self.manifest) == 0:
            raise ValueError("GAMUSDataset received an empty manifest.")

        required = {"scene"}
        missing = required - set(self.manifest.columns)
        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing)}"
            )

        self._scene_cache: OrderedDict[str, dict[str, np.ndarray]] = (
            OrderedDict()
        )

        # Deterministic validation indices: top-left aligned grid.
        self._val_locations: list[tuple[int, int, int]] = []
        if not self.training:
            self._build_validation_index()

    def __len__(self) -> int:
        if self.training:
            return len(self.manifest) * self.samples_per_scene
        return len(self._val_locations)

    def _build_validation_index(self) -> None:
        stride = self.crop_size

        for scene_index, row in self.manifest.iterrows():
            scene = str(row["scene"])
            arrays = self._load_scene(scene)

            h, w = arrays["height"].shape

            ys = list(range(0, max(1, h - self.crop_size + 1), stride))
            xs = list(range(0, max(1, w - self.crop_size + 1), stride))

            if not ys or ys[-1] != max(0, h - self.crop_size):
                ys.append(max(0, h - self.crop_size))

            if not xs or xs[-1] != max(0, w - self.crop_size):
                xs.append(max(0, w - self.crop_size))

            for y in sorted(set(ys)):
                for x in sorted(set(xs)):
                    self._val_locations.append((scene_index, y, x))

    def _scene_paths(self, scene: str) -> tuple[Path, Path, Path]:
        split_dir = self.split

        rgb = (
            self.data_root
            / "images"
            / split_dir
            / f"{scene}_RGB.h5"
        )
        agl = (
            self.data_root
            / "heights"
            / split_dir
            / f"{scene}_AGL.h5"
        )
        cls = (
            self.data_root
            / "classes"
            / split_dir
            / f"{scene}_CLS.h5"
        )

        return rgb, agl, cls

    @staticmethod
    def _read_h5(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing local GAMUS file:\n{path}"
            )

        with h5py.File(path, "r") as f:
            if "image" not in f:
                raise KeyError(
                    f"Expected 'image' dataset in {path}, "
                    f"found {list(f.keys())}"
                )
            return f["image"][:]

    def _load_scene(self, scene: str) -> dict[str, np.ndarray]:
        if scene in self._scene_cache:
            self._scene_cache.move_to_end(scene)
            return self._scene_cache[scene]

        rgb_path, agl_path, cls_path = self._scene_paths(scene)

        rgb = self._read_h5(rgb_path)
        height = self._read_h5(agl_path)

        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(
                f"{scene}: expected RGB [H,W,3], got {rgb.shape}"
            )

        if height.ndim != 2:
            raise ValueError(
                f"{scene}: expected AGL [H,W], got {height.shape}"
            )

        if rgb.shape[:2] != height.shape:
            raise ValueError(
                f"{scene}: RGB/AGL mismatch: "
                f"{rgb.shape} vs {height.shape}"
            )

        # Class data is optional for the training tensor.
        # It is loaded only when present, allowing a lighter training path.
        classes = None
        if cls_path.exists():
            classes = self._read_h5(cls_path)
            if classes.shape != height.shape:
                raise ValueError(
                    f"{scene}: CLS/AGL mismatch: "
                    f"{classes.shape} vs {height.shape}"
                )

        arrays = {
            "rgb": rgb,
            "height": height,
        }

        if classes is not None:
            arrays["class"] = classes

        self._scene_cache[scene] = arrays
        self._scene_cache.move_to_end(scene)

        while len(self._scene_cache) > self.max_cached_scenes:
            self._scene_cache.popitem(last=False)

        return arrays

    def _get_train_location(
        self,
        index: int,
        h: int,
        w: int,
    ) -> tuple[int, int]:
        # Every epoch, the same scene receives different random crops under
        # DataLoader shuffling because the RNG is based on the sample index
        # and the dataset seed.
        scene_index = index // self.samples_per_scene
        sample_index = index % self.samples_per_scene

        rng = random.Random(
            self.seed
            + scene_index * 100_003
            + sample_index * 9_973
            + torch.initial_seed() % 1_000_003
        )

        if h <= self.crop_size:
            y = 0
        else:
            y = rng.randint(0, h - self.crop_size)

        if w <= self.crop_size:
            x = 0
        else:
            x = rng.randint(0, w - self.crop_size)

        return y, x

    def _crop(
        self,
        rgb: np.ndarray,
        height: np.ndarray,
        y: int,
        x: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = height.shape

        crop_h = min(self.crop_size, h)
        crop_w = min(self.crop_size, w)

        rgb_crop = rgb[
            y:y + crop_h,
            x:x + crop_w,
        ]

        height_crop = height[
            y:y + crop_h,
            x:x + crop_w,
        ]

        # Pad smaller-than-crop scenes without creating target signal.
        if crop_h != self.crop_size or crop_w != self.crop_size:
            padded_rgb = np.zeros(
                (self.crop_size, self.crop_size, 3),
                dtype=rgb_crop.dtype,
            )
            padded_height = np.full(
                (self.crop_size, self.crop_size),
                np.nan,
                dtype=np.float32,
            )

            padded_rgb[:crop_h, :crop_w] = rgb_crop
            padded_height[:crop_h, :crop_w] = height_crop

            rgb_crop = padded_rgb
            height_crop = padded_height

        return rgb_crop, height_crop

    def _augment(
        self,
        rgb: np.ndarray,
        height: np.ndarray,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = random.Random(self.seed + index)

        if rng.random() < 0.5:
            rgb = np.flip(rgb, axis=1).copy()
            height = np.flip(height, axis=1).copy()

        if rng.random() < 0.5:
            rgb = np.flip(rgb, axis=0).copy()
            height = np.flip(height, axis=0).copy()

        return rgb, height

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if self.training:
            scene_index = index // self.samples_per_scene
            scene = str(self.manifest.iloc[scene_index]["scene"])
        else:
            scene_index, y, x = self._val_locations[index]
            scene = str(self.manifest.iloc[scene_index]["scene"])

        arrays = self._load_scene(scene)

        rgb = arrays["rgb"]
        height = arrays["height"].astype(np.float32, copy=False)

        h, w = height.shape

        if self.training:
            y, x = self._get_train_location(index, h, w)

        rgb_crop, height_crop = self._crop(
            rgb,
            height,
            y,
            x,
        )

        if self.augment:
            rgb_crop, height_crop = self._augment(
                rgb_crop,
                height_crop,
                index,
            )

        target, weights = build_height_targets(
            height_crop,
            self.quality_policy,
        )

        valid_mask = weights > 0.0

        image_tensor = torch.from_numpy(
            rgb_crop.transpose(2, 0, 1).copy()
        ).float() / 255.0

        height_tensor = torch.from_numpy(
            target.copy()
        ).float()

        weight_tensor = torch.from_numpy(
            weights.copy()
        ).float()

        valid_tensor = torch.from_numpy(
            valid_mask.copy()
        ).bool()

        if self.image_transform is not None:
            image_tensor = self.image_transform(
                image_tensor
            )

        return {
            "image": image_tensor,
            "height": height_tensor,
            "valid_mask": valid_tensor,
            "weight": weight_tensor,
            "scene": scene,
        }
