# bing_download.py

import os
import json
import urllib.parse
from bs4 import BeautifulSoup

from PIL import Image
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError


def validate_image(path):
    """
    尝试打开图片并加载一遍，若失败则返回 False。
    用于过滤下载后破损或错误格式的文件。
    """
    try:
        with Image.open(path) as img:
            img.verify()   # 检查是否完整
        # 再次 open 确保可 decode
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


# 伪装浏览器的请求头
HEADER = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; WOW64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/50.0.2661.102 Safari/537.36'
    )
}


# 必应图片异步加载接口模板
BING_URL = (
    "https://cn.bing.com/images/async?"
    "q={query}&first={first}&count={count}"
    "&scenario=ImageBasicHover&datsrc=N_I&layout=ColumnBased&mmasync=1"
    "&IG=0D6AD6CBAF43430EA716510A4754C951&SFX={sfx}&iid=images.5599"
)


def download_image(
    img_url,
    save_dir,
    index,
    prefix="img_",
    ext=".jpg",
    timeout=(5, 15),
    max_size_mb=8
):
    """
    用 requests 下载图片，支持超时和简单的大小限制。
    timeout = (连接超时, 读取超时)
    max_size_mb: 超过则中断（防止超大文件导致卡顿）。
    下载成功返回文件路径，失败返回 None。
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{prefix}{index}{ext}")
    max_bytes = max_size_mb * 1024 * 1024

    # 加上浏览器 User-Agent + 简单的 Referer，有助于减少部分站点的 403
    headers = HEADER.copy()
    headers.setdefault("Referer", "https://cn.bing.com/images/")

    try:
        resp = requests.get(img_url, stream=True, timeout=timeout, headers=headers)
        resp.raise_for_status()

        total = 0
        with open(filename, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                total += len(chunk)
                if total > max_bytes:
                    print(f"[download_image] 文件过大(>{max_size_mb}MB)，丢弃：{img_url}")
                    f.close()
                    os.remove(filename)
                    return None

    except requests.exceptions.Timeout:
        print(f"[download_image] 下载超时：{img_url}")
        return None
    except requests.RequestException as e:
        print(f"[download_image] 请求失败：{e}")
        return None
    except Exception as e:
        print(f"[download_image] 其它异常：{e}")
        return None

    # 如需校验图片有效性可以打开下面代码
    # if not validate_image(filename):
    #     print(f"[download_image] 无效图片，删除：{filename}")
    #     os.remove(filename)
    #     return None

    print(f"图片+1，成功保存第 {index} 张：{filename}")
    return filename


def parse_image_items(html_text):
    """
    从一页 HTML 中解析出所有图片的
    - 原图 URL
    - 标题（t 字段）
    返回：[{ "url": ..., "title": ... }, ...]
    """
    soup = BeautifulSoup(html_text, "lxml")
    links = soup.find_all("a", class_="iusc")

    items = []
    for link in links:
        m_attr = link.get("m")
        if not m_attr:
            continue
        try:
            meta = json.loads(m_attr)
        except Exception:
            # m 属性不是合法 JSON，跳过
            continue

        img_url = meta.get("murl") or meta.get("purl")
        title = meta.get("t", "")

        if not img_url:
            continue

        items.append({
            "url": img_url,
            "title": title
        })

    return items


def fetch_page_html(keyword, first, count, sfx, timeout=(5, 15)):
    """
    请求一页必应图片搜索结果，返回 HTML 文本。
    timeout = (连接超时, 读取超时)，单位秒。
    出错或超时时返回 ""，避免卡死。
    """
    query = urllib.parse.quote(keyword)
    full_url = BING_URL.format(query=query, first=first, count=count, sfx=sfx)

    try:
        resp = requests.get(full_url, headers=HEADER, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        print(f"[fetch_page_html] 超时：{full_url}")
        return ""
    except requests.RequestException as e:
        print(f"[fetch_page_html] 请求失败：{e}")
        return ""

    return resp.text


def save_metadata_json(metadata, meta_path):
    """
    安全地把 metadata 列表写入 JSON 文件：
    先写到临时文件，再原子替换，避免中途断电损坏文件。
    """
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    tmp_path = meta_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, meta_path)


def crawl_bing_images(
    keyword,
    save_dir="./imgs/hat",
    max_count=200,
    batch_size=35,
    file_prefix="img_",
    metadata_filename="metadata.json",
    global_seen_urls=None,     # ⭐ 跨关键词全局 URL 去重集合（本次进程共享）
    keyword_labels=None,       # 关键词标签
    resume=True,               # 已有 metadata 则跳过
    max_workers=5              # 并行线程数
):
    """
    根据单个关键词从必应图片下载图像，并保存元信息到 JSON。

    特性：
    - resume=True 且 save_dir 下存在 metadata.json → 直接跳过本关键词
    - 使用 global_seen_urls 在“本次进程内”对 URL 全局去重（跨关键词）
    - 每一页内部使用线程池并行下载，每张图最多等待 15 秒
    - 结束条件：
        * 连续若干页 HTML 请求失败
        * 连续若干页「有候选但全失败」
        * 连续若干页「候选为空（全部为已见/失败 URL）」→ 防止因全是旧图卡死
        * 页数达到上限
    """
    if keyword_labels is None:
        keyword_labels = []

    if global_seen_urls is None:
        global_seen_urls = set()

    meta_path = os.path.join(save_dir, metadata_filename)

    # ========= 1. 断点续跑：已有 metadata 就直接跳过 =========
    if resume and os.path.exists(meta_path):
        print(f"[关键词: {keyword}] 已存在 {metadata_filename}，跳过此关键词。")
        return

    metadata = []
    count = 0              # 已经“占用编号”的图片数量（本关键词）
    sfx = 1

    # HTML 请求失败页数
    fail_pages = 0
    MAX_FAIL_PAGES = 6

    # 有候选但全失败（403/超时等）的“无新图页”
    no_new_pages = 0
    MAX_NO_NEW_PAGES = 6

    # 全部为已见/失败 URL 的页数（跨关键词全局去重导致）
    old_only_pages = 0
    MAX_OLD_ONLY_PAGES = 12   # 连续这么多页都是旧图就结束该关键词

    # 一般来说翻几百页已经远远超过需要了，防止极端情况死循环
    MAX_PAGES_PER_KEYWORD = 200

    # 使用 page_index 控制分页，而不是依赖 count
    page_index = 0         # 当前页（从 0 开始）

    # 记录哪些 URL 已经尝试过但失败了，避免无限重试 403/404
    failed_urls = set()

    # ========= 2. 主循环：翻页下载 =========
    while count < max_count and page_index < MAX_PAGES_PER_KEYWORD:
        first = page_index * batch_size + 1

        # 2.1 请求一页 HTML（带超时）
        html = fetch_page_html(keyword, first, batch_size, sfx)
        if not html:
            print(f"[关键词: {keyword}] 本页 HTML 获取失败，跳过到下一页。")
            fail_pages += 1
            if fail_pages >= MAX_FAIL_PAGES:
                print(f"[关键词: {keyword}] 连续 {MAX_FAIL_PAGES} 页 HTML 失败，结束本关键词。")
                break
            page_index += 1
            sfx += 1
            continue

        # HTML 正常，重置失败计数
        fail_pages = 0

        # 2.2 解析出图片列表
        items = parse_image_items(html)
        if not items:
            print(f"[关键词: {keyword}] 本页未解析到任何图片，结束本关键词。")
            break

        # 2.3 按 URL 去重（跨关键词 + 当前关键词失败 URL）
        candidates = []
        for item in items:
            img_url = item["url"]
            if img_url in global_seen_urls:
                continue
            if img_url in failed_urls:
                continue
            candidates.append(item)

        # ⭐ 情况 A：本页全部为已见/失败 URL → 只有旧图，简单翻页，但有限次数
        if not candidates:
            old_only_pages += 1
            print(
                f"[关键词: {keyword}] 本页全部为已见/失败 URL（跨关键词全局去重），"
                f"连续旧图页数 = {old_only_pages}，global_seen_urls={len(global_seen_urls)}"
            )
            if old_only_pages >= MAX_OLD_ONLY_PAGES:
                print(
                    f"[关键词: {keyword}] 连续 {MAX_OLD_ONLY_PAGES} 页都是旧图，"
                    f"判断该关键词无明显新内容，结束本关键词。"
                )
                break
            page_index += 1
            sfx += 1
            continue
        else:
            # 一旦本页出现了至少一个“潜在新 URL”，说明我们已经翻到了新的区域，
            # 后续再遇到旧图需要重新累计。
            old_only_pages = 0

        # 2.4 并行下载这一页里的候选图片
        new_this_page = 0
        os.makedirs(save_dir, exist_ok=True)

        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for item in candidates:
                if count >= max_count:
                    break

                img_url = item["url"]
                title = item["title"]

                index = count + 1
                # 即使下载失败，这个 index 也算占用，保持编号单调
                count += 1

                future = executor.submit(
                    download_image,
                    img_url,
                    save_dir,
                    index,
                    file_prefix
                )
                tasks.append((future, img_url, title, index))

            # 收集结果，每张图最多等 15 秒
            for future, img_url, title, index in tasks:
                try:
                    filename = future.result(timeout=15)
                except TimeoutError:
                    print(f"[关键词: {keyword}] 第 {index} 张下载超时 (>15s)，跳过。")
                    failed_urls.add(img_url)
                    continue
                except Exception as e:
                    print(f"[关键词: {keyword}] 第 {index} 张下载线程异常：{e}")
                    failed_urls.add(img_url)
                    continue

                if not filename:
                    # 下载失败 / 校验失败，标记为失败 URL，之后不再尝试
                    failed_urls.add(img_url)
                    continue

                # 成功：加入跨关键词去重集合 + 记录 metadata
                global_seen_urls.add(img_url)
                record = {
                    "index": index,
                    "keyword": keyword,
                    "keyword_labels": keyword_labels,
                    "title": title,
                    "image_url": img_url,
                    "local_path": filename
                }
                metadata.append(record)
                save_metadata_json(metadata, meta_path)
                new_this_page += 1

        # 2.5 情况 B：有候选但本页所有下载都失败（403 / 超时等），算“无新图页”
        if new_this_page == 0:
            no_new_pages += 1
            print(f"[关键词: {keyword}] 本页未产生任何新图片（可能全部 403/失败），连续无新图页数 = {no_new_pages}")
            if no_new_pages >= MAX_NO_NEW_PAGES:
                print(f"[关键词: {keyword}] 连续 {MAX_NO_NEW_PAGES} 页无新图，结束本关键词。")
                break
        else:
            no_new_pages = 0

        # 2.6 当前页处理完，翻到下一页
        page_index += 1
        sfx += 1

    # ========= 3. 总结输出 =========
    print(
        f"任务结束（关键词: {keyword}），共成功下载 {len(metadata)} 张图片。"
        f" metadata 保存在 {meta_path}"
    )


if __name__ == "__main__":
    crawl_bing_images(
        keyword="Carbon Fiber",
        save_dir="./imgs/carbon_fiber",
        max_count=500,           # 期望最多下载多少张
        batch_size=35,
        file_prefix="carbon_fiber_",
        metadata_filename="carbon_fiber_metadata.json"
    )
