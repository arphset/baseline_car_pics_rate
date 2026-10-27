"""
Запуск бэйзлайна из командной строки.

    python run_baseline.py фото.jpg
    python run_baseline.py папка_с_фото --mode car --out results

Печатает таблицу по каждому фото и сохраняет:
  report.csv   — короткий отчёт с русскими колонками (открывается в Excel),
  results.csv  — все метрики,
  results.json — всё, включая детекции YOLO.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from quality_baseline import analyze

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CSV_FIELDS = [
    "file", "width", "height", "score", "verdict", "technically_ok", "main_photo_ok",
    "sharpness_score", "display_laplacian_var", "display_laplacian_local", "is_blurry", "is_soft", "is_motion_blur", "acf_score", "laplacian_var", "median_laplacian", "sobel_var",
    "brightness", "overexposed_pct", "underexposed_pct", "is_overexposed", "is_underexposed", "is_dark", "is_bright",
    "object_found", "main_class", "object_count", "fill_rate", "is_off_center",
    "car_found", "car_fill", "car_cropped", "other_large_cars", "extra_objects",
    "issues", "context_issues", "penalties", "time_s",
]


def _yes(v):
    return "да" if v else "нет"


def write_report(rows, path, mode):
    """Короткая таблица с русскими колонками — для чтения глазами. Все метрики — в results.csv."""
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        wr = csv.writer(fh, delimiter=";")
        head = ["Файл", "Разрешение", "Балл", "Вердикт", "Технически годно"]
        if mode == "car":
            head += ["Годится на главное", "Машина найдена", "Класс YOLO", "Доля кадра под машиной, %",
                     "Машина обрезана", "Лишние объекты"]
        head += ["Индекс резкости", "Проблемы", "Штрафы"]
        wr.writerow(head)
        for r in rows:
            if "error" in r:
                wr.writerow([r["file"], "", "", "ошибка: " + r["error"]])
                continue
            row = [r["file"], f"{r['width']}x{r['height']}", r["score"], r["verdict"], _yes(r["technically_ok"])]
            if mode == "car":
                row += [_yes(r["main_photo_ok"]), _yes(r["car_found"]), r.get("main_class") or "",
                        round(r["car_fill"] * 100), _yes(r["car_cropped"]),
                        ", ".join(f"{k}: {v}" for k, v in r["extra_objects"].items())]
            row += [round(r["sharpness_score"], 2), "; ".join(r["issues"] + r.get("context_issues", [])),
                    "; ".join(r["penalties"])]
            wr.writerow(row)


def collect(path):
    p = Path(path)
    if p.is_dir():
        return sorted((f for f in p.iterdir() if f.suffix.lower() in EXTS),
                      key=lambda f: (len(f.stem), f.stem))
    return [p]


def main():
    ap = argparse.ArgumentParser(description="Эвристический бэйзлайн оценки качества изображений")
    ap.add_argument("path", help="файл или папка с изображениями")
    ap.add_argument("--mode", choices=["generic", "car"], default="generic",
                    help="generic — любые картинки; car — плюс проверки для главного фото автомобиля")
    ap.add_argument("--profile", choices=["revised", "original"], default="revised",
                    help="revised — исправленная оценка; original — формулы и пороги как в QualityJPG")
    ap.add_argument("--out", default="results", help="папка для results.csv и results.json")
    ap.add_argument("--quiet", action="store_true", help="не печатать таблицу")
    args = ap.parse_args()

    files = collect(args.path)
    if not files:
        sys.exit(f"Нет изображений: {args.path}")

    rows = []
    for f in files:
        r = analyze(f, mode=args.mode, profile=args.profile)
        r["file"] = f.name
        rows.append(r)
        if args.quiet:
            continue
        if "error" in r:
            print(f"{f.name:<24} ОШИБКА: {r['error']}")
            continue
        line = f"{f.name:<24} {r['score']:>4}  {r['verdict']:<18}"
        if args.mode == "car":
            line += f" главное фото: {'да ' if r['main_photo_ok'] else 'нет'}"
        reasons = r["issues"] + r.get("context_issues", [])
        print(line + ("  | " + "; ".join(reasons) if reasons else ""))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    write_report(rows, out / "report.csv", args.mode)
    # utf-8-sig, чтобы Excel правильно открыл кириллицу
    with open(out / "results.csv", "w", encoding="utf-8-sig", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore", delimiter=";")
        wr.writeheader()
        for r in rows:
            r = dict(r)
            for k in ("issues", "context_issues", "penalties"):
                if isinstance(r.get(k), list):
                    r[k] = "; ".join(r[k])
            if isinstance(r.get("extra_objects"), dict):
                r["extra_objects"] = ", ".join(f"{k}: {v}" for k, v in r["extra_objects"].items())
            wr.writerow(r)

    ok = [r for r in rows if "error" not in r]
    if ok:
        print(f"\nФото: {len(ok)}, средний балл: {sum(r['score'] for r in ok) / len(ok):.1f}, "
              f"технически годных: {sum(r['technically_ok'] for r in ok)}"
              + (f", годных на главное: {sum(r['main_photo_ok'] for r in ok)}" if args.mode == "car" else ""))
        print(f"Среднее время на фото: {sum(r['time_s'] for r in ok) / len(ok):.2f} с")
    print(f"Результаты: {out / 'report.csv'}, {out / 'results.csv'}, {out / 'results.json'}")


if __name__ == "__main__":
    main()
