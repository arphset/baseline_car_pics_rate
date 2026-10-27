"""
Сравнение вердикта бэйзлайна «годится на главное фото» с ручной разметкой.

    python evaluate.py results/results.json labels.json

labels.json: {"img_4.jpg": ["3/4 спереди", 1], ...} — тип кадра и метка 1 (годится) / 0 (не годится).
"""
import json
import sys
from collections import Counter


def main(results_path, labels_path):
    rows = json.load(open(results_path, encoding="utf-8"))
    labels = json.load(open(labels_path, encoding="utf-8"))
    tp = fp = tn = fn = 0
    errors = []
    fp_by_type = Counter()
    for r in rows:
        if r["file"] not in labels or "error" in r:
            continue
        frame, y = labels[r["file"]]
        p = int(r["main_photo_ok"])
        tp += p and y
        fp += p and not y
        tn += (not p) and (not y)
        fn += (not p) and y
        if p != y:
            if p:
                fp_by_type[frame] += 1
            errors.append(f"  {r['file']:<12} {frame:<24} бэйзлайн {'принял' if p else 'отклонил'}"
                          f"  ({'; '.join(r['issues'] + r.get('context_issues', [])) or 'проблем не найдено'})")
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"Фото: {n}   TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Accuracy {(tp + tn) / n:.2f}   Precision {prec:.2f}   Recall {rec:.2f}   F1 {f1:.2f}")
    if fp_by_type:
        print("Ошибочно принятые по типу кадра:", dict(fp_by_type))
    print("Ошибки:")
    print("\n".join(errors) or "  нет")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
