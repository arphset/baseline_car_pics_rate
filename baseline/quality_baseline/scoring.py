"""Итоговый технический балл по 10-балльной шкале (логика score_calculator.py из QualityJPG)."""
from .config import BLUR_REVISED, PROFILE, RESOLUTION, SCORE, SOFT_PENALTY


def quality_score(r, cfg=SCORE, profile=None):
    """
    Исправления относительно QualityJPG:
    - пересвет: раньше за один и тот же яркий кадр штрафовали дважды (is_overexposed −1.5 и is_bright −1.5),
      теперь — один раз;
    - смаз движения уже снижает индекс резкости (×0.7); отдельный штраф −1.5 теперь не добавляется,
      если фото и так получило полный штраф за размытие.
    """
    score = 10.0
    penalties = []

    s = r.get("sharpness_score", 1.0)
    blur_full = s < 0.50
    if blur_full:
        score -= cfg["blur_full"]
        penalties.append(f"Размытие: -{cfg['blur_full']}")
    elif r.get("is_soft"):
        score -= SOFT_PENALTY
        penalties.append(f"Нечёткое (мыльное) изображение: -{SOFT_PENALTY}")
    elif s < cfg["blur_soft_from"]:
        # в revised плавный штраф остаётся только для диапазона 0.6–0.7 (максимум −1.0)
        lo = BLUR_REVISED["soft_threshold"] if (profile or PROFILE) == "revised" else 0.50
        top = 1.0 if (profile or PROFILE) == "revised" else cfg["blur_soft_max"]
        p = round(top * (cfg["blur_soft_from"] - s) / (cfg["blur_soft_from"] - lo), 1)
        score -= p
        penalties.append(f"Пограничная чёткость: -{p}")

    if r.get("is_motion_blur") and not blur_full:
        score -= cfg["motion"]
        penalties.append(f"Смаз движения: -{cfg['motion']}")

    if r.get("is_dark"):
        score -= cfg["dark"]
        penalties.append(f"Слишком темно: -{cfg['dark']}")
    elif r.get("is_underexposed"):
        score -= cfg["underexposed"]
        penalties.append(f"Недосвет: -{cfg['underexposed']}")

    if r.get("is_overexposed") or r.get("is_bright"):
        p = max(cfg["overexposed"], cfg["bright"])
        score -= p
        penalties.append(f"Пересвет ({r.get('overexposed_pct', 0):.1f}%): -{p}")

    fill = float(r.get("fill_rate") or 0)
    if not r.get("object_found", True):
        score -= cfg["not_found"]
        penalties.append(f"Объект не найден: -{cfg['not_found']}")
    elif fill < 0.10:
        score -= cfg["fill_lt10"]
        penalties.append(f"Заполнение {fill * 100:.0f}%: -{cfg['fill_lt10']}")
    elif fill < 0.20:
        score -= cfg["fill_lt20"]
        penalties.append(f"Заполнение {fill * 100:.0f}%: -{cfg['fill_lt20']}")
    elif fill < 0.30:
        score -= cfg["fill_lt30"]
        penalties.append(f"Заполнение {fill * 100:.0f}%: -{cfg['fill_lt30']}")

    if r.get("object_found", True) and r.get("is_off_center"):
        score -= cfg["off_center"]
        penalties.append(f"Смещение от центра: -{cfg['off_center']}")

    if (profile or PROFILE) == "revised" and r.get("width"):
        short = min(r["width"], r["height"])
        if short < RESOLUTION["very_low"]:
            score -= RESOLUTION["penalty_very_low"]
            penalties.append(f"Очень низкое разрешение {r['width']}x{r['height']}: -{RESOLUTION['penalty_very_low']}")
        elif short < RESOLUTION["low"]:
            score -= RESOLUTION["penalty_low"]
            penalties.append(f"Низкое разрешение {r['width']}x{r['height']}: -{RESOLUTION['penalty_low']}")
        elif short < RESOLUTION["medium"]:
            score -= RESOLUTION["penalty_medium"]
            penalties.append(f"Разрешение ниже рекомендуемого {r['width']}x{r['height']}: -{RESOLUTION['penalty_medium']}")

    score = round(max(1.0, min(10.0, score)), 1)
    verdict = next(v for thr, v in cfg["verdicts"] if score >= thr)
    has_defect = any(r.get(k) for k in ("is_blurry", "is_soft", "is_motion_blur", "is_dark", "is_underexposed",
                                         "is_overexposed", "is_bright"))
    ok = score >= cfg["good_threshold"] and not (cfg["defect_fails"] and has_defect)
    return {"score": score, "verdict": verdict, "penalties": penalties, "technically_ok": ok}
