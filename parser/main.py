"""Парсер фото объявлений Авто.ру в исходном разрешении.

Исправление относительно main_original_backup.py:
- раньше ссылки брались из <img src> на странице: это превью 584x438
  (и 'small'), а замена размера в fix_url не срабатывала, потому что
  URL кончается на '/584x438' без слеша в конце;
- теперь список фото берётся из данных объявления (state.image_urls),
  и для каждого фото качается версия /orig (исходник, обычно 3-4K);
- файл пишется байт в байт, без пересжатия, с расширением по Content-Type;
- рядом сохраняется meta.json: ракурс (photo_class), исходный размер и URL.

Капча и вход не обходятся: если сайт отдаёт капчу или редирект на логин,
парсер останавливается. С флагом --browser страницы открываются в Chrome,
и капчу при необходимости решает человек (фото всё равно качаются с CDN).
"""
import json
from html import unescape
import os
import random
import re
import sys
import time

import requests

OUTPUT_DIR = "dataset_auto_ru"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

PAUSE_BETWEEN_LISTINGS = (8, 12)   # секунды
PAUSE_BETWEEN_PHOTOS = (0.5, 1.5)

EXT_BY_TYPE = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


class BlockedError(Exception):
    """Сайт показал капчу или потребовал вход."""


session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ru-RU,ru;q=0.9"})


driver = None   # Selenium, только в режиме --browser


def is_blocked(url, text):
    return ("showcaptcha" in url or "auth.auto.ru" in url
            or "smartcaptcha" in text[:20000])


def fetch_listing_html(listing_url):
    if driver is not None:
        # Капчу решает человек в открытом окне, парсер только ждёт
        driver.get(listing_url)
        while is_blocked(driver.current_url, driver.page_source):
            input("🧩 Решите капчу в окне браузера и нажмите ENTER (Ctrl+C - выход)... ")
        time.sleep(2)
        # После загрузки страницы скрипт сайта удаляет из DOM данные объявления,
        # поэтому сырой HTML берём обычным запросом с cookies этого браузера
        for c in driver.get_cookies():
            session.cookies.set(c["name"], c["value"], domain=c.get("domain"))
        resp = session.get(listing_url, timeout=20)
        if not is_blocked(resp.url, resp.text) and '"image_urls"' in resp.text:
            return resp.text
        return driver.page_source
    resp = session.get(listing_url, timeout=20)
    if is_blocked(resp.url, resp.text):
        raise BlockedError(f"Капча/логин на {listing_url} -> {resp.url}")
    resp.raise_for_status()
    return resp.text


def extract_images(html):
    """Достаёт массив image_urls из встроенного JSON карточки объявления."""
    i = html.find('"image_urls":')
    if i < 0:
        html = unescape(html)   # данные могут лежать в атрибуте с &quot;
        i = html.find('"image_urls":')
    if i < 0:
        return extract_gallery_images(html)
    start = html.index("[", i)
    images, _ = json.JSONDecoder().raw_decode(html, start)
    return images


def extract_gallery_images(html):
    """Запасной путь для страницы из Chrome: фото из галереи объявления.

    В отрисованной странице JSON с image_urls уже удалён, но галерея
    (блок ImageGalleryDesktop) содержит ссылки на все фото объявления.
    Ракурса и исходного размера тут нет.
    """
    a, b = html.find("ImageGalleryDesktop"), html.rfind("ImageGalleryDesktop")
    if a < 0:
        return []
    ids = []
    for m in re.finditer(r"(//avatars\.(?:avto\.ru|mds\.yandex\.net)/get-autoru-vos/\d+/[0-9a-f]+)/",
                         html[a:b + 3000]):
        if m.group(1) not in ids:
            ids.append(m.group(1))
    return [{"sizes": {"1200x900": u + "/1200x900"}} for u in ids]


def orig_url(sizes):
    """Из любой ссылки на размер делает ссылку на исходник (/orig)."""
    url = sizes.get("1200x900") or next(iter(sizes.values()))
    if url.startswith("//"):
        url = "https:" + url
    return re.sub(r"/[^/]+$", "/orig", url), url


def download(url, save_path_no_ext):
    resp = session.get(url, headers={"Referer": "https://auto.ru/"}, timeout=60)
    ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if resp.status_code != 200 or not ctype.startswith("image/"):
        return None
    path = save_path_no_ext + EXT_BY_TYPE.get(ctype, ".jpg")
    with open(path, "wb") as f:   # без декодирования и пересжатия
        f.write(resp.content)
    return path


def parse_and_download_photos(listing_url):
    print(f"\n🚗 {listing_url}")
    html = fetch_listing_html(listing_url)
    images = extract_images(html)
    sale_id = re.search(r"/(\d+-[0-9a-f]+)/?$", listing_url).group(1)
    if not images:
        dump = f"debug_{sale_id}.html"
        with open(dump, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"   ⚠️ Фото не найдены (объявление снято?), страница сохранена в {dump}")
        return 0

    car_dir = os.path.join(OUTPUT_DIR, sale_id)
    os.makedirs(car_dir, exist_ok=True)

    meta = {"url": listing_url, "photos": []}
    for n, img in enumerate(images, 1):
        url, fallback = orig_url(img["sizes"])
        base = os.path.join(car_dir, f"img_{n:02d}")
        path = download(url, base)
        if path is None:   # исходник недоступен, берём крупнейшее превью
            url = fallback
            path = download(url, base)
        if path:
            meta["photos"].append({
                "file": os.path.basename(path),
                "source_url": url,
                "photo_class": img.get("photo_class"),
                "orig_width": img.get("orig_width"),
                "orig_height": img.get("orig_height"),
            })
            print(f"    ✅ {os.path.basename(path)} ({os.path.getsize(path) // 1024} KB)")
        time.sleep(random.uniform(*PAUSE_BETWEEN_PHOTOS))

    with open(os.path.join(car_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"   🎉 Сохранено {len(meta['photos'])} из {len(images)} -> {car_dir}")
    return len(meta["photos"])


def main(urls):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🚀 Объявлений: {len(urls)}")
    for k, url in enumerate(urls):
        try:
            parse_and_download_photos(url)
        except BlockedError as e:
            print(f"⛔ {e}\nОстанавливаюсь, защиту сайта не обходим.")
            return 1
        except requests.RequestException as e:
            print(f"   ❌ {e}")
        if k < len(urls) - 1:
            time.sleep(random.uniform(*PAUSE_BETWEEN_LISTINGS))
    print(f"\n✅ Готово, папка '{OUTPUT_DIR}'")
    return 0


def start_browser():
    from selenium import webdriver
    opts = webdriver.ChromeOptions()
    opts.add_argument(f"user-agent={USER_AGENT}")
    return webdriver.Chrome(options=opts)


if __name__ == "__main__":
    # python main.py urls.txt            — файл со ссылками, по одной в строке
    # python main.py urls.txt --browser  — через Chrome, капчу решаете вы
    args = [a for a in sys.argv[1:] if a != "--browser"]
    if "--browser" in sys.argv:
        driver = start_browser()
    if args:
        with open(args[0], encoding="utf-8") as f:
            urls = [u.strip() for u in f if u.strip()]
    else:
        urls = [
            "https://auto.ru/cars/used/sale/kia/sorento/1132130352-04ec82fe/",
            "https://auto.ru/cars/used/sale/volvo/xc60/1126346229-1e72d47c/",
        ]
    try:
        code = main(urls)
    finally:
        if driver is not None:
            driver.quit()
    sys.exit(code)
