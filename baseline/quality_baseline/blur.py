"""Оценка резкости: ACF, лапласиан (глобальный и по блокам), градиент Собеля, смаз движения."""
import cv2
import numpy as np

from .config import BLUR, BLUR_REVISED, PROFILE


def check_blur(image, roi=None, cfg=BLUR, profile=None):
    """
    image — BGR-изображение, roi — (x1, y1, x2, y2) области объекта или None.

    Отличие от QualityJPG: раньше фон вне рамки заливался серым 128, а метрики считались
    по всему кадру. На границе заливки возникал искусственный резкий перепад, который
    завышал лапласиан и Собель. Теперь метрики считаются по вырезанной области объекта.
    """
    profile = profile or PROFILE
    full_h, full_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))  # яркость всего кадра — как в исходной версии

    if roi is not None:
        x1, y1, x2, y2 = roi
        crop = gray[y1:y2, x1:x2]
        # слишком маленькая рамка не даёт надёжной оценки — берём весь кадр
        if crop.shape[0] >= 32 and crop.shape[1] >= 32:
            gray = crop
        else:
            roi = None
    h, w = gray.shape
    gray_f = gray.astype(np.float64)

    # ACF: насколько пиксели похожи на соседей через lag пикселей (у размытого фото — сильно)
    norm = (gray_f - gray_f.mean()) / (gray_f.std() + 1e-8)
    fft = np.fft.fft2(norm)
    acf = np.fft.fftshift(np.fft.ifft2(fft * np.conj(fft)).real)
    cy, cx = h // 2, w // 2
    acf = acf / (acf[cy, cx] + 1e-8)
    lag = cfg["acf_lag"]
    acf_lag = (abs(acf[cy - lag, cx]) + abs(acf[cy, cx - lag])) / 2

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    bs = cfg["block_size"]
    local = [cv2.Laplacian(gray[y:y + bs, x:x + bs], cv2.CV_64F).var()
             for y in range(0, h - bs + 1, bs) for x in range(0, w - bs + 1, bs)]
    median_laplacian = float(np.median(local)) if local else laplacian_var

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_var = float(np.var(np.sqrt(sobelx ** 2 + sobely ** 2)))

    # Смаз движения: градиенты сильно преобладают в одном направлении + высокая автокорреляция
    gx, gy = np.mean(np.abs(sobelx)), np.mean(np.abs(sobely))
    direction_ratio = max(gx, gy) / (min(gx, gy) + 1e-8)
    is_motion_blur = bool(direction_ratio > cfg["motion_direction_ratio"] and acf_lag > cfg["motion_acf"])

    if profile == "revised":
        sharpness, perceived = _perceived_sharpness(gray, full_h, full_w)
        if is_motion_blur:
            sharpness *= cfg["motion_factor"]
            method = "смаз движения"
        else:
            method = "резкость при показе 1024 px"
    elif brightness < cfg["dark_brightness"] or brightness > cfg["bright_brightness"]:
        dark = brightness < cfg["dark_brightness"]
        lap_s = min(1.0, laplacian_var / (cfg["norm_lap_dark"] if dark else cfg["norm_lap_bright"]))
        sob_s = min(1.0, sobel_var / (cfg["norm_sobel_dark"] if dark else cfg["norm_sobel_bright"]))
        wl, ws = cfg["weights_extreme"]
        sharpness = wl * lap_s + ws * sob_s
        method = "тёмное фото" if dark else "светлое фото"
    else:
        scores = (
            max(0.0, 1.0 - acf_lag / cfg["norm_acf"]),
            min(1.0, laplacian_var / cfg["norm_lap_global"]),
            min(1.0, median_laplacian / cfg["norm_lap_local"]),
            min(1.0, sobel_var / cfg["norm_sobel"]),
        )
        sharpness = float(np.dot(cfg["weights"], scores))
        if is_motion_blur:
            sharpness *= cfg["motion_factor"]
            method = "смаз движения"
        else:
            method = "обычное фото"
    if profile != "revised":
        perceived = {}
    if roi is not None:
        method += " (по области объекта)"

    return {
        "acf_score": float(acf_lag),
        "laplacian_var": laplacian_var,
        "median_laplacian": median_laplacian,
        "sobel_var": sobel_var,
        "sharpness_score": float(sharpness),
        "is_blurry": bool(sharpness < cfg["blurry_threshold"]),
        "is_soft": bool(profile == "revised" and cfg["blurry_threshold"] <= sharpness < BLUR_REVISED["soft_threshold"]),
        "is_motion_blur": is_motion_blur,
        "blur_method": method,
        **perceived,
    }


def _perceived_sharpness(gray, full_h, full_w, cfg=BLUR_REVISED):
    """
    Резкость на стандартном размере показа. Масштаб считается по всему кадру, а не по области
    объекта, чтобы объект увеличивался ровно так же, как на экране покупателя.
    """
    scale = cfg["ref_size"] / max(full_h, full_w)
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp).astype(np.float64)
    std = g.std()
    if std < cfg["low_contrast_std"]:
        g = (g - g.mean()) * (cfg["contrast_target_std"] / (std + 1e-6)) + g.mean()
    lap = cv2.Laplacian(g, cv2.CV_64F)
    lap_var = float(lap.var())
    bs = cfg["block_size"]
    blocks = [lap[y:y + bs, x:x + bs].var()
              for y in range(0, g.shape[0] - bs + 1, bs) for x in range(0, g.shape[1] - bs + 1, bs)]
    lap_local = float(np.percentile(blocks, cfg["local_percentile"])) if blocks else lap_var

    def to_unit(v):
        return float(np.clip(np.log(max(v, 1e-6) / cfg["log_lo"]) / np.log(cfg["log_hi"] / cfg["log_lo"]), 0, 1))

    sharpness = (to_unit(lap_var) + to_unit(lap_local)) / 2
    return sharpness, {"display_laplacian_var": lap_var, "display_laplacian_local": lap_local}
