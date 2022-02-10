""" download data and create toutch dataset """
from typing import (Tuple, Any, List, Optional, Callable)

import logging
import coloredlogs
import h5py as h5
import json

from pathlib import Path
from PIL import Image

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (check_integrity, download_url)

class MoonCraterDataset(VisionDataset):
    """ Moon crater Data.
    
        Learning with craters. This module will make the data available
    """

    logger = logging.getLogger(__name__)

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
        loglevel: str = "DEBUG"
    ) -> None:
        super().__init__(
            root, transform=transform, target_transform=target_transform
        )

        coloredlogs.install(level=loglevel, logger=self.logger)

        self.root = Path(self.root)
        self.root.mkdir(exist_ok=True)

        # if you want to download the data
        if download:
            self.logger.info("start download")
            self.download()

        # the data is not available
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. \n You can use download=True to download it")

        self.data_file = h5.File(self.root / "moon_data.h5", mode="r")
        self.crater_info = None

        with open(self.root / "data_rec.json", "r", encoding="utf8") as jsonfile:
            self.logger.info("read crater info")
            # the data is a list
            crater_data = tuple(json.load(jsonfile))
            # more easy access to the data information. 
            # the data is not stored randomly in the datafile. to find the right 
            # crater information in the crater_data list, build a hashmap
            self.crater_info = { c_data["name"]: c_data for c_data in crater_data }
    
    def __len__(self) -> int:
        """ get the number of samples
        
        Return:
            int: number of samples
        """
        return self.data_file["/image"].shape[0]

    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """ get the sample of index
                
        Args:
            index (int): index of the next element

        Returns:
            Tuple[Any,Any,Any]: return a sample
        """

        # image and target
        img, target = Image.fromarray(self.data_file["/image"][index,...]), Image.fromarray(self.data_file["/mask"][index,...])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # get the crater information
        crater = self.crater_info[str(self.data_file["/names"][index])]

        return img, target, crater


    def __del__(self):
        """ close the datafile, when the objects is deleted """
        self.data_file.close()

    def _check_integrity(self) -> bool:
        """ check if the files are available and the md5
            sum is valid
        
        Returns:
            bool: the files are valid
        """
        for fentry in self.file_list:
            md5, filename = fentry[0], fentry[1]
            self.logger.debug(f"check file {filename} with md5 hash {md5}")
            if not check_integrity(fpath=self.root/filename, md5=md5):
                return False
        return True

    def download(self) -> None:
        """ start the download process """

        # check if the files are not already there.
        # carefull, this can take a while, because the
        # md5 validation
        if self._check_integrity():
            self.logger.info("Files already downloaded and verified")
            return

        # the download
        for fentry in self.file_list:
            file_name = fentry[1]
            self.logger.warn("start download url")
            download_url(f"{self.url}/{file_name}", str(self.root), filename=file_name, md5=fentry[0])
