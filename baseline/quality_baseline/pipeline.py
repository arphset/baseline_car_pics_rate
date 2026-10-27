"""Единая точка входа бэйзлайна: analyze(path или изображение) → словарь с метриками, баллом и причинами."""
import time

import cv2
import numpy as np

from .blur import check_blur
from .config import PROFILE, RESOLUTION
from .detection import car_context, composition, detect
from .exposure import check_exposure
from .scoring import quality_score


def load_image(path):
    # imdecode вместо imread: imread в Windows не открывает пути с кириллицей
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def analyze(image_or_path, mode="generic", profile=None):
    """
    mode="generic" — техническая оценка любой картинки (группа А);
    mode="car"     — дополнительно контекстные проверки автомобиля (часть группы Б)
                     и итог «годится на главное фото» по эвристикам.
    profile — "revised" (по умолчанию) или "original" (формулы и пороги QualityJPG), см. config.py.
    """
    t0 = time.perf_counter()
    image = image_or_path if isinstance(image_or_path, np.ndarray) else load_image(image_or_path)
    if image is None:
        return {"error": "Не удалось открыть изображение"}
    h, w = image.shape[:2]

    detections = detect(image)
    comp = composition(image, detections, mode=mode)
    blur = check_blur(image, roi=comp["roi"], profile=profile)
    expo = check_exposure(image, profile)

    result = {"width": w, "height": h, "mode": mode, "profile": profile or PROFILE}
    result.update(blur)
    result.update(expo)
    result.update({k: v for k, v in comp.items() if k != "roi"})
    result["detections"] = [{"cls": d["cls"], "conf": round(d["conf"], 3), "box": d["box"]} for d in detections]

    issues = []
    if result["profile"] == "revised" and min(w, h) < RESOLUTION["low"]:
        issues.append(f"Низкое разрешение ({w}x{h})")
    if blur["is_blurry"]:
        issues.append("Размыто")
    if blur["is_soft"]:
        issues.append("Нечёткое (мыльное) изображение")
    if blur["is_motion_blur"]:
        issues.append("Смаз движения")
    if expo["is_overexposed"] or expo["is_bright"]:
        issues.append("Пересвет")
    if expo["is_dark"]:
        issues.append("Слишком темно")
    elif expo["is_underexposed"]:
        issues.append("Недосвет")
    issues.extend(comp["composition_issues"])
    result["issues"] = issues

    result.update(quality_score(result, profile=profile))

    if mode == "car":
        ctx = car_context(image, detections)
        result.update(ctx)
        # Итог по эвристикам: технически годно и нет контекстных проблем.
        # Тип кадра (сбоку / багажник / салон) бэйзлайн не проверяет — это задача классификатора.
        result["main_photo_ok"] = bool(result["technically_ok"] and not ctx["context_issues"])

    result["time_s"] = round(time.perf_counter() - t0, 3)
    return result
