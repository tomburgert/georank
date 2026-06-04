from typing import Tuple
from pydantic import Field, validator, validate_arguments

import numpy as np

from bigearthnet_patch_interface.band_interface import Band
from bigearthnet_patch_interface.patch_interface import BigEarthNetPatch


class SeCo_10mBand(Band):
    name: str
    spatial_resolution: int = 10
    data_shape: Tuple[int, int] = (264, 264)

    @validator("name")
    def _validate_name(cls, v):
        ben10m_bands = ("B02", "B03", "B04", "B08")
        if v not in ben10m_bands:
            raise ValueError(f"Band name must one of {ben10m_bands}\nGiven: {v}")
        return v


class SeCo_20mBand(Band):
    spatial_resolution: int = 20
    data_shape: Tuple[int, int] = (132, 132)

    @validator("name")
    def _validate_name(cls, v):
        ben20m_bands = ("B05", "B06", "B07", "B8A", "B11", "B12")
        if v not in ben20m_bands:
            raise ValueError(f"Band name must one of {ben20m_bands}\nGiven: {v}")
        return v


class SeCo_60mBand(Band):
    spatial_resolution: int = 60
    data_shape: Tuple[int, int] = (44, 44)

    @validator("name")
    def _validate_name(cls, v):
        ben20m_bands = ("B01", "B09")
        if v not in ben20m_bands:
            raise ValueError(f"Band name must one of {ben20m_bands}\nGiven: {v}")
        return v


class SeCo_Patch(BigEarthNetPatch):
    def __init__(
        self,
        band01: np.ndarray,
        band02: np.ndarray,
        band03: np.ndarray,
        band04: np.ndarray,
        band05: np.ndarray,
        band06: np.ndarray,
        band07: np.ndarray,
        band08: np.ndarray,
        band8A: np.ndarray,
        band09: np.ndarray,
        band11: np.ndarray,
        band12: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Original BigEarthNet-S2 patch class.
        Will store any additional keyword arguments as attributes
        and also in `__stored_args__`.

        Args:
            band01 (np.ndarray): 60m band
            band02 (np.ndarray): 10m band
            band03 (np.ndarray): 10m band
            band04 (np.ndarray): 10m band
            band05 (np.ndarray): 20m band
            band06 (np.ndarray): 20m band
            band07 (np.ndarray): 20m band
            band08 (np.ndarray): 10m band
            band8A (np.ndarray): 20m band
            band09 (np.ndarray): 60m band
            band11 (np.ndarray): 20m band
            band12 (np.ndarray): 20m band
        """
        self.band01 = SeCo_60mBand(name="B01", data=band01)
        self.band02 = SeCo_10mBand(name="B02", data=band02)
        self.band03 = SeCo_10mBand(name="B03", data=band03)
        self.band04 = SeCo_10mBand(name="B04", data=band04)
        self.band05 = SeCo_20mBand(name="B05", data=band05)
        self.band06 = SeCo_20mBand(name="B06", data=band06)
        self.band07 = SeCo_20mBand(name="B07", data=band07)
        self.band08 = SeCo_10mBand(name="B08", data=band08)
        self.band8A = SeCo_20mBand(name="B8A", data=band8A)
        self.band09 = SeCo_60mBand(name="B09", data=band09)
        self.band11 = SeCo_20mBand(name="B11", data=band11)
        self.band12 = SeCo_20mBand(name="B12", data=band12)

        bands = [
            self.band01,
            self.band02,
            self.band03,
            self.band04,
            self.band05,
            self.band06,
            self.band07,
            self.band08,
            self.band8A,
            self.band09,
            self.band11,
            self.band12,
        ]
        super().__init__(bands=bands, **kwargs)

    @staticmethod
    @validate_arguments
    def short_to_long_band_name(
        short_band_name: str = Field(..., min_length=1, max_length=4)
    ) -> str:
        """
        Convert a short input band name, such as B01, 01, 1 or B1
        to the long band name required by the `__init__` function.

        Args:
            short_band_name (str): Short band name
        """
        short_upper = short_band_name.upper()
        band = short_upper.lstrip("B")
        if band.endswith("A"):
            num = int(band.rstrip("A"))
            # only Band8A should have an A at the end!
            # for compability, it should _not_ include 08A
            band = f"{num:01d}A"
        else:
            num = int(band)
            band = f"{num:02d}"
        return f"band{band}"

    @classmethod
    def short_init(
        cls,
        B01: np.ndarray,
        B02: np.ndarray,
        B03: np.ndarray,
        B04: np.ndarray,
        B05: np.ndarray,
        B06: np.ndarray,
        B07: np.ndarray,
        B08: np.ndarray,
        B8A: np.ndarray,
        B09: np.ndarray,
        B11: np.ndarray,
        B12: np.ndarray,
        **kwargs,
    ) -> "SeCo_Patch":
        """
        Alternative `__init__` function.
        Only differences are the encoded names.
        """
        return cls(
            band01=B01,
            band02=B02,
            band03=B03,
            band04=B04,
            band05=B05,
            band06=B06,
            band07=B07,
            band08=B08,
            band8A=B8A,
            band09=B09,
            band11=B11,
            band12=B12,
            **kwargs,
        )

    def get_10m_bands(self) -> Tuple[np.ndarray, ...]:
        """
        Get the _data_ of the 10m bands of the patch interface as a tuple
        The ordering is given by `self.bands`.
        The ordering is guaranteed to be naturally sorted:
        B02, B03, B04, B08

        Returns:
            Tuple[np.ndarray]: Bands: B02, B03, B04, B08
        """
        return super().get_natsorted_bands_by_spatial_res(spatial_resolution=10)

    def get_20m_bands(self) -> Tuple[np.ndarray, ...]:
        """
        Get the _data_ of the 20m bands of the patch interface as a tuple
        The ordering is given by `self.bands`.
        The ordering is guaranteed to be naturally sorted:
        B05, B06, B07, B8A, B11, B12

        Returns:
            Tuple[np.ndarray]: Bands: B05, B06, B07, B8A, B11, B12
        """
        return super().get_natsorted_bands_by_spatial_res(spatial_resolution=20)

    def get_60m_bands(self) -> Tuple[np.ndarray, ...]:
        """
        Get  the _data_ of the 60m bands of the patch interface as a tuple
        The ordering is given by `self.bands`.
        The ordering is guaranteed to be naturally sorted:
        B01, B09

        Returns:
            Tuple[np.ndarray]: Bands: B01, B09
        """
        return super().get_natsorted_bands_by_spatial_res(spatial_resolution=60)

    def get_stacked_10m_bands(self) -> np.ndarray:
        """
        Quick way to get the 10m bands already stacked.
        Calls `np.stack(self.get_10m_bands())`.

        Returns:
            np.ndarray: Stacked 10m bands
        """
        return np.stack(self.get_10m_bands())

    def get_stacked_20m_bands(self) -> np.ndarray:
        """
        Quick way to get the 20m bands already stacked.
        Calls `np.stack(self.get_20m_bands())`.

        Returns:
            np.ndarray: Stacked 20m bands
        """
        return np.stack(self.get_20m_bands())

    def get_stacked_60m_bands(self) -> np.ndarray:
        """
        Quick way to get the 60m bands already stacked.
        Calls `np.stack(self.get_60m_bands())`.

        Returns:
            np.ndarray: Stacked 60m bands
        """
        return np.stack(self.get_60m_bands())


class S4A_10mBand(Band):
    name: str
    spatial_resolution: int = 10
    data_shape: Tuple[int, int] = (122, 122)

    @validator("name")
    def _validate_name(cls, v):
        ben10m_bands = ("B02", "B03", "B04", "B08")
        if v not in ben10m_bands:
            raise ValueError(f"Band name must one of {ben10m_bands}\nGiven: {v}")
        return v


class S4A_20mBand(Band):
    spatial_resolution: int = 20
    data_shape: Tuple[int, int] = (61, 61)

    @validator("name")
    def _validate_name(cls, v):
        ben20m_bands = ("B05", "B06", "B07", "B8A", "B11", "B12")
        if v not in ben20m_bands:
            raise ValueError(f"Band name must one of {ben20m_bands}\nGiven: {v}")
        return v


class S4A_Patch(BigEarthNetPatch):
    def __init__(
        self,
        band02: np.ndarray,
        band03: np.ndarray,
        band04: np.ndarray,
        band05: np.ndarray,
        band06: np.ndarray,
        band07: np.ndarray,
        band08: np.ndarray,
        band8A: np.ndarray,
        band11: np.ndarray,
        band12: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Original BigEarthNet-S2 patch class.
        Will store any additional keyword arguments as attributes
        and also in `__stored_args__`.

        Args:
            band01 (np.ndarray): 60m band
            band02 (np.ndarray): 10m band
            band03 (np.ndarray): 10m band
            band04 (np.ndarray): 10m band
            band05 (np.ndarray): 20m band
            band06 (np.ndarray): 20m band
            band07 (np.ndarray): 20m band
            band08 (np.ndarray): 10m band
            band8A (np.ndarray): 20m band
            band09 (np.ndarray): 60m band
            band11 (np.ndarray): 20m band
            band12 (np.ndarray): 20m band
        """
        self.band02 = S4A_10mBand(name="B02", data=band02)
        self.band03 = S4A_10mBand(name="B03", data=band03)
        self.band04 = S4A_10mBand(name="B04", data=band04)
        self.band05 = S4A_20mBand(name="B05", data=band05)
        self.band06 = S4A_20mBand(name="B06", data=band06)
        self.band07 = S4A_20mBand(name="B07", data=band07)
        self.band08 = S4A_10mBand(name="B08", data=band08)
        self.band8A = S4A_20mBand(name="B8A", data=band8A)
        self.band11 = S4A_20mBand(name="B11", data=band11)
        self.band12 = S4A_20mBand(name="B12", data=band12)

        bands = [
            self.band02,
            self.band03,
            self.band04,
            self.band05,
            self.band06,
            self.band07,
            self.band08,
            self.band8A,
            self.band11,
            self.band12,
        ]
        super().__init__(bands=bands, **kwargs)

    @staticmethod
    @validate_arguments
    def short_to_long_band_name(
        short_band_name: str = Field(..., min_length=1, max_length=4)
    ) -> str:
        """
        Convert a short input band name, such as B01, 01, 1 or B1
        to the long band name required by the `__init__` function.

        Args:
            short_band_name (str): Short band name
        """
        short_upper = short_band_name.upper()
        band = short_upper.lstrip("B")
        if band.endswith("A"):
            num = int(band.rstrip("A"))
            # only Band8A should have an A at the end!
            # for compability, it should _not_ include 08A
            band = f"{num:01d}A"
        else:
            num = int(band)
            band = f"{num:02d}"
        return f"band{band}"

    @classmethod
    def short_init(
        cls,
        B02: np.ndarray,
        B03: np.ndarray,
        B04: np.ndarray,
        B05: np.ndarray,
        B06: np.ndarray,
        B07: np.ndarray,
        B08: np.ndarray,
        B8A: np.ndarray,
        B11: np.ndarray,
        B12: np.ndarray,
        **kwargs,
    ) -> "S4A_Patch":
        """
        Alternative `__init__` function.
        Only differences are the encoded names.
        """
        return cls(
            band02=B02,
            band03=B03,
            band04=B04,
            band05=B05,
            band06=B06,
            band07=B07,
            band08=B08,
            band8A=B8A,
            band11=B11,
            band12=B12,
            **kwargs,
        )

    def get_10m_bands(self) -> Tuple[np.ndarray, ...]:
        """
        Get the _data_ of the 10m bands of the patch interface as a tuple
        The ordering is given by `self.bands`.
        The ordering is guaranteed to be naturally sorted:
        B02, B03, B04, B08

        Returns:
            Tuple[np.ndarray]: Bands: B02, B03, B04, B08
        """
        return super().get_natsorted_bands_by_spatial_res(spatial_resolution=10)

    def get_20m_bands(self) -> Tuple[np.ndarray, ...]:
        """
        Get the _data_ of the 20m bands of the patch interface as a tuple
        The ordering is given by `self.bands`.
        The ordering is guaranteed to be naturally sorted:
        B05, B06, B07, B8A, B11, B12

        Returns:
            Tuple[np.ndarray]: Bands: B05, B06, B07, B8A, B11, B12
        """
        return super().get_natsorted_bands_by_spatial_res(spatial_resolution=20)

    def get_stacked_10m_bands(self) -> np.ndarray:
        """
        Quick way to get the 10m bands already stacked.
        Calls `np.stack(self.get_10m_bands())`.

        Returns:
            np.ndarray: Stacked 10m bands
        """
        return np.stack(self.get_10m_bands())

    def get_stacked_20m_bands(self) -> np.ndarray:
        """
        Quick way to get the 20m bands already stacked.
        Calls `np.stack(self.get_20m_bands())`.

        Returns:
            np.ndarray: Stacked 20m bands
        """
        return np.stack(self.get_20m_bands())


class AnyS2_Patch(BigEarthNetPatch):
    def __init__(
        self,
        band01: np.ndarray,
        band02: np.ndarray,
        band03: np.ndarray,
        band04: np.ndarray,
        band05: np.ndarray,
        band06: np.ndarray,
        band07: np.ndarray,
        band08: np.ndarray,
        band8A: np.ndarray,
        band09: np.ndarray,
        band11: np.ndarray,
        band12: np.ndarray,
        band10_shape: Tuple[int, int],
        band20_shape: Tuple[int, int], 
        band60_shape: Tuple[int, int], 
        **kwargs,
    ) -> None:
        """
        Original BigEarthNet-S2 patch class.
        Will store any additional keyword arguments as attributes
        and also in `__stored_args__`.

        Args:
            band01 (np.ndarray): 60m band
            band02 (np.ndarray): 10m band
            band03 (np.ndarray): 10m band
            band04 (np.ndarray): 10m band
            band05 (np.ndarray): 20m band
            band06 (np.ndarray): 20m band
            band07 (np.ndarray): 20m band
            band08 (np.ndarray): 10m band
            band8A (np.ndarray): 20m band
            band09 (np.ndarray): 60m band
            band11 (np.ndarray): 20m band
            band12 (np.ndarray): 20m band
        """
        self.band01 = SeCo_60mBand(name="B01", data=band01, data_shape=band60_shape)
        self.band02 = SeCo_10mBand(name="B02", data=band02, data_shape=band10_shape)
        self.band03 = SeCo_10mBand(name="B03", data=band03, data_shape=band10_shape)
        self.band04 = SeCo_10mBand(name="B04", data=band04, data_shape=band10_shape)
        self.band05 = SeCo_20mBand(name="B05", data=band05, data_shape=band20_shape)
        self.band06 = SeCo_20mBand(name="B06", data=band06, data_shape=band20_shape)
        self.band07 = SeCo_20mBand(name="B07", data=band07, data_shape=band20_shape)
        self.band08 = SeCo_10mBand(name="B08", data=band08, data_shape=band10_shape)
        self.band8A = SeCo_20mBand(name="B8A", data=band8A, data_shape=band20_shape)
        self.band09 = SeCo_60mBand(name="B09", data=band09, data_shape=band60_shape)
        self.band11 = SeCo_20mBand(name="B11", data=band11, data_shape=band20_shape)
        self.band12 = SeCo_20mBand(name="B12", data=band12, data_shape=band20_shape)

        bands = [
            self.band01,
            self.band02,
            self.band03,
            self.band04,
            self.band05,
            self.band06,
            self.band07,
            self.band08,
            self.band8A,
            self.band09,
            self.band11,
            self.band12,
        ]
        super().__init__(bands=bands, **kwargs)

    @staticmethod
    @validate_arguments
    def short_to_long_band_name(
        short_band_name: str = Field(..., min_length=1, max_length=4)
    ) -> str:
        """
        Convert a short input band name, such as B01, 01, 1 or B1
        to the long band name required by the `__init__` function.

        Args:
            short_band_name (str): Short band name
        """
        short_upper = short_band_name.upper()
        band = short_upper.lstrip("B")
        if band.endswith("A"):
            num = int(band.rstrip("A"))
            # only Band8A should have an A at the end!
            # for compability, it should _not_ include 08A
            band = f"{num:01d}A"
        else:
            num = int(band)
            band = f"{num:02d}"
        return f"band{band}"

    @classmethod
    def short_init(
        cls,
        B01: np.ndarray,
        B02: np.ndarray,
        B03: np.ndarray,
        B04: np.ndarray,
        B05: np.ndarray,
        B06: np.ndarray,
        B07: np.ndarray,
        B08: np.ndarray,
        B8A: np.ndarray,
        B09: np.ndarray,
        B11: np.ndarray,
        B12: np.ndarray,
        **kwargs,
    ) -> "SeCo_Patch":
        """
        Alternative `__init__` function.
        Only differences are the encoded names.
        """
        return cls(
            band01=B01,
            band02=B02,
            band03=B03,
            band04=B04,
            band05=B05,
            band06=B06,
            band07=B07,
            band08=B08,
            band8A=B8A,
            band09=B09,
            band11=B11,
            band12=B12,
            **kwargs,
        )

    def get_10m_bands(self) -> Tuple[np.ndarray, ...]:
        """
        Get the _data_ of the 10m bands of the patch interface as a tuple
        The ordering is given by `self.bands`.
        The ordering is guaranteed to be naturally sorted:
        B02, B03, B04, B08

        Returns:
            Tuple[np.ndarray]: Bands: B02, B03, B04, B08
        """
        return super().get_natsorted_bands_by_spatial_res(spatial_resolution=10)

    def get_20m_bands(self) -> Tuple[np.ndarray, ...]:
        """
        Get the _data_ of the 20m bands of the patch interface as a tuple
        The ordering is given by `self.bands`.
        The ordering is guaranteed to be naturally sorted:
        B05, B06, B07, B8A, B11, B12

        Returns:
            Tuple[np.ndarray]: Bands: B05, B06, B07, B8A, B11, B12
        """
        return super().get_natsorted_bands_by_spatial_res(spatial_resolution=20)

    def get_60m_bands(self) -> Tuple[np.ndarray, ...]:
        """
        Get  the _data_ of the 60m bands of the patch interface as a tuple
        The ordering is given by `self.bands`.
        The ordering is guaranteed to be naturally sorted:
        B01, B09

        Returns:
            Tuple[np.ndarray]: Bands: B01, B09
        """
        return super().get_natsorted_bands_by_spatial_res(spatial_resolution=60)

    def get_stacked_10m_bands(self) -> np.ndarray:
        """
        Quick way to get the 10m bands already stacked.
        Calls `np.stack(self.get_10m_bands())`.

        Returns:
            np.ndarray: Stacked 10m bands
        """
        return np.stack(self.get_10m_bands())

    def get_stacked_20m_bands(self) -> np.ndarray:
        """
        Quick way to get the 20m bands already stacked.
        Calls `np.stack(self.get_20m_bands())`.

        Returns:
            np.ndarray: Stacked 20m bands
        """
        return np.stack(self.get_20m_bands())

    def get_stacked_60m_bands(self) -> np.ndarray:
        """
        Quick way to get the 60m bands already stacked.
        Calls `np.stack(self.get_60m_bands())`.

        Returns:
            np.ndarray: Stacked 60m bands
        """
        return np.stack(self.get_60m_bands())
