"""Оценка экспозиции по гистограмме: доля пересвеченных и провалившихся в тень пикселей, средняя яркость."""
import cv2
import numpy as np

from .config import EXPOSURE_PROFILE, EXPOSURE_PROFILES


def check_exposure(image, profile=None):
    cfg = EXPOSURE_PROFILES[profile or EXPOSURE_PROFILE]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))

    # адаптивные пороги для тёмных (low-key) и светлых (high-key) кадров — только в исходном профиле
    if cfg["adaptive"] and brightness < cfg["lowkey_brightness"]:
        over_thr, under_thr = cfg["over_thr_lowkey"], cfg["under_thr_lowkey"]
    elif cfg["adaptive"] and brightness > cfg["highkey_brightness"]:
        over_thr, under_thr = cfg["over_thr_highkey"], cfg["under_thr_highkey"]
    else:
        over_thr, under_thr = cfg["over_thr_normal"], cfg["under_thr_normal"]

    over_pct = float(np.mean(gray > over_thr) * 100)
    under_pct = float(np.mean(gray < under_thr) * 100)

    return {
        "brightness": brightness,
        "overexposed_pct": over_pct,
        "underexposed_pct": under_pct,
        "is_overexposed": over_pct > cfg["overexposed_pct"],
        "is_underexposed": under_pct > cfg["underexposed_pct"],
        "is_dark": brightness < cfg["dark_brightness"],
        "is_bright": brightness > cfg["bright_brightness"],
        "exposure_profile": profile or EXPOSURE_PROFILE,
    }
