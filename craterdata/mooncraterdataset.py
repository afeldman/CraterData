"""PyTorch Dataset for crater analysis — supports local HDF5 and ldm-client backends."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import coloredlogs
import h5py as h5
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_url


class MoonCraterDataset(VisionDataset):
    """Moon crater dataset for machine learning.

    Unterstützt zwei Datenquellen:

    1. **Lokale HDF5-Dateien** (bisheriges Verhalten) – lädt ``moon_data.h5`` +
       ``data_rec.json`` von Zenodo oder lokalem Pfad.

    2. **ldm-client / luna-data-mare** (neu) – lädt Krater aus dem Robbins-
       Katalog + NAC-Bilder + LOLA-DEMs direkt aus MinIO + PostgreSQL.
       Aktiviert durch Angabe von ``pg_url`` und ``minio_endpoint``.

    Args:
        root: Lokaler Daten-Pfad (für HDF5-Modus und Cache).
        transform: Torchvision-Transform für Bilder.
        target_transform: Transform für Targets.
        download: HDF5-Daten von Zenodo herunterladen (nur HDF5-Modus).
        loglevel: Logging-Level.
        pg_url: PostgreSQL-URL für ldm-client-Modus.
        minio_endpoint: MinIO-Endpoint für ldm-client-Modus.
        minio_access: MinIO Access Key.
        minio_secret: MinIO Secret Key.
        roi: (lat_min, lat_max, lon_min, lon_max) – räumliche Eingrenzung.
        min_diam_km: Minimale Kratergröße in km.
        max_diam_km: Maximale Kratergröße in km.
        cache_images: NAC-Bilder lokal cachen (Standard: True).
    """

    logger = logging.getLogger(__name__)

    # HDF5-Modus
    url = "https://zenodo.org/records/5563001/files"
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
        loglevel: str = "DEBUG",
        # ldm-client Parameter (optional)
        pg_url: str | None = None,
        minio_endpoint: str | None = None,
        minio_access: str = "minioadmin",
        minio_secret: str = "minioadmin",
        roi: tuple[float, float, float, float] | None = None,
        min_diam_km: float = 1.0,
        max_diam_km: float | None = None,
        cache_images: bool = True,
    ) -> None:
        super().__init__(
            root, transform=transform, target_transform=target_transform
        )

        coloredlogs.install(level=loglevel, logger=self.logger)

        self.root = Path(self.root)
        self.root.mkdir(exist_ok=True)

        # ── Modus-Auswahl ──────────────────────────────────────────
        if pg_url and minio_endpoint:
            self._init_ldm(
                pg_url=pg_url,
                minio_endpoint=minio_endpoint,
                minio_access=minio_access,
                minio_secret=minio_secret,
                roi=roi,
                min_diam_km=min_diam_km,
                max_diam_km=max_diam_km,
                cache_images=cache_images,
            )
        else:
            self._init_hdf5(download=download)

    # ── HDF5-Modus (bisherig) ────────────────────────────────────

    def _init_hdf5(self, download: bool = False) -> None:
        """Alten HDF5-Modus initialisieren (kompatibel zu v0.2.0)."""
        if download:
            self.logger.info("start download")
            self._download_hdf5()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. \n You can use download=True to download it"
            )

        self.data_file = h5.File(self.root / "moon_data.h5", mode="r")
        with open(self.root / "data_rec.json", "r", encoding="utf8") as jsonfile:
            self.logger.info("read crater info")
            crater_data = tuple(json.load(jsonfile))
            self.crater_info = {
                c_data["name"]: c_data for c_data in crater_data
            }

    def _check_integrity(self) -> bool:
        for md5, filename in self.file_list:
            self.logger.debug(f"check file {filename} with md5 hash {md5}")
            if not check_integrity(fpath=self.root / filename, md5=md5):
                return False
        return True

    def _download_hdf5(self) -> None:
        if self._check_integrity():
            self.logger.info("Files already downloaded and verified")
            return
        for md5, file_name in self.file_list:
            self.logger.warning("start download url")
            download_url(
                f"{self.url}/{file_name}", str(self.root),
                filename=file_name, md5=md5,
            )

    def _getitem_hdf5(self, index: int):
        img = Image.fromarray(self.data_file["/image"][index, ...])
        target = Image.fromarray(self.data_file["/mask"][index, ...])

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        crater = self.crater_info[str(self.data_file["/names"][index])]
        return img, target, crater

    # ── ldm-client-Modus (neu) ────────────────────────────────────

    def _init_ldm(
        self,
        pg_url: str,
        minio_endpoint: str,
        minio_access: str,
        minio_secret: str,
        roi: tuple[float, float, float, float] | None,
        min_diam_km: float,
        max_diam_km: float | None,
        cache_images: bool,
    ) -> None:
        """Krater + Bilder via ldm-client aus luna-data-mare laden."""
        from ldm_client import PluginRegistry

        self._pg_url = pg_url
        self._minio_endpoint = minio_endpoint
        self._minio_access = minio_access
        self._minio_secret = minio_secret
        self._cache_images = cache_images

        # Plugins initialisieren
        self._robbins_plugin = PluginRegistry.get("robbins")(
            pg_url=pg_url,
        )
        self._lro_plugin = PluginRegistry.get("lro")()
        self._lola_plugin = PluginRegistry.get("lola")()

        # Kraterliste laden
        self.logger.info("Lade Krater aus Robbins-Katalog ...")
        import psycopg

        conditions = []
        params: dict[str, Any] = {}
        if roi is not None:
            lat_min, lat_max, lon_min, lon_max = roi
            conditions.append("lat BETWEEN %(lat_min)s AND %(lat_max)s")
            conditions.append("lon BETWEEN %(lon_min)s AND %(lon_max)s")
            params.update(lat_min=lat_min, lat_max=lat_max,
                          lon_min=lon_min, lon_max=lon_max)
        conditions.append("diameter_km >= %(min_diam)s")
        params["min_diam"] = min_diam_km
        if max_diam_km is not None:
            conditions.append("diameter_km <= %(max_diam)s")
            params["max_diam"] = max_diam_km

        where = " AND ".join(conditions)
        with psycopg.connect(pg_url) as conn:
            rows = conn.execute(
                f"SELECT crater_id, lat, lon, diameter_km, depth_m, morphology "
                f"FROM robbins_craters WHERE {where} "
                f"ORDER BY diameter_km DESC",
                params,
            ).fetchall()

        self._craters = [
            {
                "crater_id": r[0],
                "lat": float(r[1]),
                "lon": float(r[2]),
                "diameter_km": float(r[3]),
                "depth_m": float(r[4]) if r[4] is not None else None,
                "morphology": r[5],
            }
            for r in rows
        ]
        self.logger.info(f"{len(self._craters)} Krater geladen")

        # Lokaler Cache für NAC-Bilder
        if self._cache_images:
            self._cache_dir = self.root / ".ldm_cache" / "images"
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _getitem_ldm(self, index: int):
        crater = self._craters[index]
        lat, lon = crater["lat"], crater["lon"]

        # NAC-Bild laden (via Plugin → MinIO)
        img = self._lro_plugin.get_image(
            lat, lon,
            pg_url=self._pg_url,
            minio_endpoint=self._minio_endpoint,
            minio_access=self._minio_access,
            minio_secret=self._minio_secret,
        )
        if img is None:
            # Fallback: leeres Bild zurückgeben
            img = np.zeros((256, 256), dtype=np.float32)

        # DEM dazuladen (für spätere Nutzung)
        dem = self._lola_plugin.get_tile(
            lat, lon,
            pg_url=self._pg_url,
            minio_endpoint=self._minio_endpoint,
            minio_access=self._minio_access,
            minio_secret=self._minio_secret,
        )

        img_pil = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
        if self.transform is not None:
            img_pil = self.transform(img_pil)

        # Metadaten als Ziel
        target = {
            "crater_id": crater["crater_id"],
            "lat": crater["lat"],
            "lon": crater["lon"],
            "diameter_km": crater["diameter_km"],
            "depth_m": crater["depth_m"],
            "morphology": crater["morphology"],
            "has_dem": dem is not None,
        }
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_pil, target

    # ── Gemeinsame API ────────────────────────────────────────────

    def __len__(self) -> int:
        if hasattr(self, "_craters"):
            return len(self._craters)
        return self.data_file["/image"].shape[0]

    def __getitem__(self, index: int):
        if hasattr(self, "_craters"):
            return self._getitem_ldm(index)
        return self._getitem_hdf5(index)

    def __del__(self):
        if hasattr(self, "data_file"):
            self.data_file.close()
