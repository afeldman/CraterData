#!/usr/bin/env python3
import sys
import fire

from craterdata.mooncraterdataset import MoonCraterDataset

def download(filepath:str):

    MoonCraterDataset(filepath, download=True)

sys.exit(fire.Fire(download))