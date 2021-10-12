import os
from typing import (Tuple, Any, List, Optional, Callable)


import h5py as h5
import json

from pathlib import Path
from PIL import Image
import numpy as np

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (check_integrity, download_url)

class MoonCraterDataset(VisionDataset):

    url='https://zenodo.org/record/5563001/files/'
    file_list = [
        ("9aa79078ec762aaabe524107e55f5328", "moon_data.h5"),
        ("066c1c44c046ae1e9722987f88edc062", "data_rec.json"),
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.root = Path(self.root)
        self.root.mkdir(exist_ok=True)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")

        self.data_file = h5.File(self.root/"moon_data.h5")
        self.crater_info = None

        with open(self.root / "data_rec.json", "r", encoding="utf8") as jsonfile:
            self.crater_info = tuple(json.load(jsonfile))
    
    def __len__(self) -> int:
        return len(self.crater_info)

    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = self.data_file["/image"][index,...], self.data_file["/mask"][index,...]

        img, target = Image.fromarray(img), Image.fromarray(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        crater = self.crater_info[int(self.data_file["/names"][index])]

        return img, target, crater


    def __del__(self):
        self.data_file.close()

    def _check_integrity(self) -> bool:
        for fentry in self.file_list:
            md5, filename = fentry[0], fentry[1]
            if not check_integrity(self.root / filename, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for fentry in self.file_list:
            file_name = fentry[1]
            download_url(f"{self.url}/{file_name}", str(self.root), filename=file_name, md5=fentry[0])
