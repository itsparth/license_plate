from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, Field

# 29 Indian state/UT codes (as of 2025)
STATE_CODES = [
    "AN",  # Andaman and Nicobar Islands
    "AP",  # Andhra Pradesh
    "AR",  # Arunachal Pradesh
    "AS",  # Assam
    "BR",  # Bihar
    "CH",  # Chandigarh
    "CG",  # Chhattisgarh
    "DD",  # Daman and Diu / Dadra and Nagar Haveli
    "DL",  # Delhi
    "GA",  # Goa
    "GJ",  # Gujarat
    "HP",  # Himachal Pradesh
    "HR",  # Haryana
    "JH",  # Jharkhand
    "JK",  # Jammu and Kashmir
    "KA",  # Karnataka
    "KL",  # Kerala
    "LA",  # Ladakh
    "LD",  # Lakshadweep
    "MH",  # Maharashtra
    "ML",  # Meghalaya
    "MN",  # Manipur
    "MP",  # Madhya Pradesh
    "MZ",  # Mizoram
    "NL",  # Nagaland
    "OD",  # Odisha
    "PB",  # Punjab
    "PY",  # Puducherry
    "RJ",  # Rajasthan
    "SK",  # Sikkim
    "TN",  # Tamil Nadu
    "TR",  # Tripura
    "TS",  # Telangana
    "UK",  # Uttarakhand
    "UP",  # Uttar Pradesh
    "WB",  # West Bengal
]

# Letters excluding O and I (confusable with 0 and 1)
PLATE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
PLATE_DIGITS = "0123456789"


class IndianLicensePlate(BaseModel):
    state_code: str = Field(..., min_length=2, max_length=2)
    district_code: int = Field(..., ge=0, le=99)
    letters: str = Field(..., min_length=1, max_length=3)
    digits: str = Field(..., min_length=4, max_length=4)
    zero_pad_district: bool = True
    is_bharat_series: bool = False

    @property
    def district_formatted(self) -> str:
        """Format district code with optional zero-padding"""
        return (
            f"{self.district_code:02d}"
            if self.zero_pad_district
            else str(self.district_code)
        )

    @property
    def formatted(self) -> str:
        if self.is_bharat_series:
            # Bharat series format: YYBH####XX (e.g., 24BH1234AB)
            return (
                f"{self.district_formatted}{self.state_code}{self.digits}{self.letters}"
            )
        return f"{self.state_code}{self.district_formatted}{self.letters}{self.digits}"

    @property
    def characters(self) -> list[str]:
        return list(self.formatted)

    @property
    def character_types(self) -> list[Literal["letter", "digit"]]:
        return ["letter" if c.isalpha() else "digit" for c in self.formatted]


class PlateGenerator:
    @staticmethod
    def random_state() -> str:
        return random.choice(STATE_CODES)

    @staticmethod
    def random_district() -> int:
        return random.randint(0, 99)

    @staticmethod
    def random_letters(*, length: int | None = None) -> str:
        if length is None:
            length = random.randint(1, 3)
        return "".join(random.choices(PLATE_LETTERS, k=length))

    @staticmethod
    def random_digits() -> str:
        return "".join(random.choices(PLATE_DIGITS, k=4))

    @staticmethod
    def random_year() -> int:
        """Random year for Bharat series (22-99 representing 2022-2099)"""
        return random.randint(22, 99)

    @staticmethod
    def random_series() -> str:
        """Random 2-letter series code for Bharat series"""
        return "".join(random.choices(PLATE_LETTERS, k=2))

    @staticmethod
    def generate(
        *,
        state_code: str | None = None,
        district_code: int | None = None,
        letters: str | None = None,
        digits: str | None = None,
        zero_pad_district: bool | None = None,
        is_bharat_series: bool = False,
    ) -> IndianLicensePlate:
        if is_bharat_series:
            state_code = "BH"
            if letters is None:
                letters = PlateGenerator.random_series()  # 2 letters for BH
            if district_code is None:
                district_code = PlateGenerator.random_year()  # Year as district for BH
            if zero_pad_district is None:
                zero_pad_district = True  # Always pad for BH
        else:
            if state_code is None:
                state_code = PlateGenerator.random_state()
            if district_code is None:
                district_code = PlateGenerator.random_district()
            if letters is None:
                letters = PlateGenerator.random_letters()
            if zero_pad_district is None:
                zero_pad_district = random.choice([True, False])

        if digits is None:
            digits = PlateGenerator.random_digits()

        return IndianLicensePlate(
            state_code=state_code,
            district_code=district_code,
            letters=letters,
            digits=digits,
            zero_pad_district=zero_pad_district,
            is_bharat_series=is_bharat_series,
        )


__all__ = [
    "IndianLicensePlate",
    "PlateGenerator",
    "STATE_CODES",
    "PLATE_LETTERS",
    "PLATE_DIGITS",
]
