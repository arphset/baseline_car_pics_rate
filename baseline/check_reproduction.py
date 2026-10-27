"""Проверка воспроизводимости результатов курсовой (раздел 2.4).

Заново прогоняет базовое решение по всем объявлениям из ../dataset
и сравнивает балл и решения с сохранёнными results/dataset/<объявление>/results.json.

    python check_reproduction.py            # все 8 объявлений, 140 фото
    python check_reproduction.py --dataset путь/к/dataset
"""
import argparse
import json
import sys
from pathlib import Path

from quality_baseline import analyze

HERE = Path(__file__).resolve().parent
FIELDS = ("score", "technically_ok", "main_photo_ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=str(HERE.parent / "dataset"))
    ap.add_argument("--results", default=str(HERE / "results" / "dataset"))
    args = ap.parse_args()

    total = mismatches = accepted = 0
    for saved_path in sorted(Path(args.results).glob("*/results.json")):
        listing = saved_path.parent.name
        saved = json.load(open(saved_path, encoding="utf-8"))
        for row in saved:
            img = Path(args.dataset) / listing / row["file"]
            r = analyze(img, mode="car")
            total += 1
            accepted += bool(r.get("main_photo_ok"))
            diff = [f"{k}: {row.get(k)} -> {r.get(k)}" for k in FIELDS if row.get(k) != r.get(k)]
            if diff:
                mismatches += 1
                print(f"{listing}/{row['file']}: " + "; ".join(diff))
        print(f"{listing}: проверено")

    print(f"\nФото: {total}, принято на главное: {accepted}, расхождений: {mismatches}")
    sys.exit(1 if mismatches or total == 0 else 0)


if __name__ == "__main__":
    main()
