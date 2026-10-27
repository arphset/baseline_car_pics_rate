"""
Детекция объектов YOLOv8 и признаки композиции.

Два режима:
- generic — как в QualityJPG: главный объект любой класс COCO (самая крупная рамка);
- car — главный объект автомобиль, плюс контекстные проверки группы Б из плана (п. 2.1):
  есть ли автомобиль, не обрезан ли он, нет ли людей и лишних объектов.
"""
import os
import threading

import numpy as np

from .config import CAR, CAR_CLASSES, DETECTION, EXTRA_OBJECTS

_model = None
_lock = threading.Lock()


def get_model(weights=None):
    """Модель загружается один раз на процесс. Если весов нет рядом, ultralytics скачает их сам."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from ultralytics import YOLO
                _model = YOLO(weights or os.environ.get("YOLO_WEIGHTS") or DETECTION["weights"])
    return _model


def detect(image, cfg=DETECTION):
    """Возвращает список детекций: класс, уверенность, рамка (x1, y1, x2, y2)."""
    model = get_model()
    with _lock:  # предиктор ultralytics не потокобезопасен
        res = model(image, conf=cfg["conf"], verbose=False)[0]
    names = res.names
    out = []
    for box in res.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
        out.append({"cls": names[int(box.cls[0])], "conf": float(box.conf[0]), "box": (x1, y1, x2, y2)})
    return out


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (_area(a) + _area(b) - inter + 1e-8)


def _union_fill(boxes, h, w):
    """
    Доля кадра под объектами. В QualityJPG площади рамок складывались, поэтому
    при пересекающихся рамках «заполнение» могло быть больше 100 %. Здесь — площадь объединения.
    """
    if not boxes:
        return 0.0
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    return float(mask.mean())


def _padded(box, h, w, pad):
    x1, y1, x2, y2 = box
    return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)


def composition(image, detections, mode="generic", cfg=DETECTION):
    h, w = image.shape[:2]
    if mode == "car":
        main = [d for d in detections if d["cls"] in CAR_CLASSES and d["conf"] >= CAR["min_car_conf"]]
    else:
        main = detections
    boxes = [d["box"] for d in main]

    largest = max(main, key=lambda d: _area(d["box"])) if main else None
    object_found = largest is not None
    max_conf = largest["conf"] if largest else 0.0
    fill_rate = _union_fill(boxes, h, w)

    # центр группы объектов, как в исходной версии
    if boxes:
        gcx = np.mean([(b[0] + b[2]) / 2 for b in boxes])
        gcy = np.mean([(b[1] + b[3]) / 2 for b in boxes])
    else:
        gcx, gcy = w / 2, h / 2
    is_off_center = bool(abs(gcx - w / 2) > w * cfg["center_tolerance"]
                         or abs(gcy - h / 2) > h * cfg["center_tolerance"])

    issues = []
    if not object_found or (fill_rate < cfg["not_found_fill"] and max_conf < cfg["not_found_conf"]):
        object_found = False
        issues.append("Автомобиль не найден" if mode == "car" else "Объект не найден")
    elif fill_rate < cfg["fill_critical"]:
        issues.append(f"Объект занимает всего {fill_rate * 100:.0f}% кадра (критически мало)")
    elif fill_rate < cfg["fill_recommended"]:
        issues.append(f"Объект занимает {fill_rate * 100:.0f}% кадра (рекомендуется > 30%)")
    if object_found and is_off_center:
        issues.append("Объект смещён от центра")

    return {
        "object_found": object_found,
        "object_count": len(main),
        "main_class": largest["cls"] if largest else None,
        "max_confidence": max_conf,
        "fill_rate": fill_rate if object_found else 0.0,
        "is_off_center": is_off_center,
        "roi": _padded(largest["box"], h, w, cfg["bbox_padding"]) if largest else None,
        "composition_issues": issues,
    }


def car_context(image, detections, cfg=DETECTION, car_cfg=CAR):
    """
    Контекстные критерии группы Б, которые можно проверить без обучения.
    Тип кадра (сбоку / багажник / салон) отсюда определить нельзя: YOLO видит класс car в любом случае.
    """
    h, w = image.shape[:2]
    img_area = h * w
    cars = sorted((d for d in detections if d["cls"] in CAR_CLASSES and d["conf"] >= car_cfg["min_car_conf"]),
                  key=lambda d: _area(d["box"]), reverse=True)
    issues = []
    ctx = {"car_found": bool(cars), "car_cropped": False, "car_fill": 0.0,
           "extra_objects": {}, "other_large_cars": 0}
    if not cars:
        issues.append("На фото не найден автомобиль")
        ctx["context_issues"] = issues
        return ctx

    main = cars[0]["box"]
    ctx["car_fill"] = _area(main) / img_area
    mx, my = w * cfg["edge_margin"], h * cfg["edge_margin"]
    x1, y1, x2, y2 = main
    ctx["car_cropped"] = bool(x1 <= mx or y1 <= my or x2 >= w - mx or y2 >= h - my)
    if ctx["car_cropped"]:
        issues.append("Автомобиль обрезан краем кадра")
    if ctx["car_fill"] < car_cfg["min_fill"]:
        issues.append(f"Автомобиль занимает {ctx['car_fill'] * 100:.0f}% кадра (мало)")

    # YOLO часто находит одну и ту же машину дважды (car и truck с почти одинаковой рамкой),
    # такие дубли не считаем «другими машинами»
    others = [d for d in cars[1:] if _iou(d["box"], main) < car_cfg["duplicate_iou"]]
    ctx["other_large_cars"] = sum(1 for d in others if _area(d["box"]) > car_cfg["other_car_ratio"] * _area(main))
    if ctx["other_large_cars"]:
        issues.append("В кадре есть другие крупные автомобили")

    for group, classes in EXTRA_OBJECTS.items():
        found = [d["cls"] for d in detections
                 if d["cls"] in classes and _area(d["box"]) / img_area >= car_cfg["extra_min_area"]]
        if found:
            ctx["extra_objects"][group] = len(found)
            issues.append(f"Лишние объекты: {group} ({len(found)})")

    ctx["context_issues"] = issues
    return ctx
