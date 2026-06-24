# batch_carbon_fiber_crawl.py
import os
import time
import json
import argparse

from carbon_fiber_keywords import CARBON_FIBER_KEYWORDS
from bing_download import crawl_bing_images


def slugify(text: str, max_len: int = 40) -> str:
    """
    把关键词转成适合做文件夹/前缀的“安全字符串”：
    - 全小写
    - 空格变下划线
    - 去掉奇怪符号
    """
    import re
    text = text.strip().lower()
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9_\u4e00-\u9fa5]", "", text)
    if len(text) > max_len:
        text = text[:max_len]
    if not text:
        text = "kw"
    return text


def label_keyword(kw: str):
    """
    根据关键词内容打一些粗粒度标签，用于后续训练/筛选。
    标签只是弱标签，可以多打一点。
    """
    kw_lower = kw.lower()
    labels = ["domain:carbon_fiber"]

    # 阶段标签
    if any(x in kw_lower for x in ["precursor", "前驱体"]):
        labels.append("stage:precursor")
    if any(x in kw_lower for x in ["spinning", "纺丝"]):
        labels.append("stage:spinning")
    if any(x in kw_lower for x in ["oxidation", "stabilization", "预氧化"]):
        labels.append("stage:oxidation")
    if any(x in kw_lower for x in ["carbonization", "graphitization", "碳化", "石墨化"]):
        labels.append("stage:carbonization")
    if any(x in kw_lower for x in ["surface treatment", "sizing", "上浆", "表面处理"]):
        labels.append("stage:surface_treatment")
    if any(x in kw_lower for x in ["weaving", "braiding", "织机", "编织"]):
        labels.append("stage:weaving")
    if any(x in kw_lower for x in ["prepreg", "预浸料"]):
        labels.append("stage:prepreg")
    if any(x in kw_lower for x in ["rtm", "infusion", "autoclave", "molding", "成型", "热压罐"]):
        labels.append("stage:forming")
    if any(x in kw_lower for x in ["testing", "test", "试验", "测试"]):
        labels.append("stage:testing")
    if any(x in kw_lower for x in ["ndt", "non destructive", "ct scan", "x ray", "红外", "无损"]):
        labels.append("stage:ndt")
    if any(x in kw_lower for x in ["sem", "micrograph", "显微", "断面"]):
        labels.append("stage:microstructure")

    # 应用领域标签
    if any(x in kw_lower for x in ["aircraft", "aerospace", "satellite", "rocket", "飞机", "航天"]):
        labels.append("app:aerospace")
    if any(x in kw_lower for x in ["automotive", "car", "motorcycle", "汽车", "摩托"]):
        labels.append("app:automotive")
    if any(x in kw_lower for x in ["wind turbine", "blade", "风电"]):
        labels.append("app:energy")
    if any(x in kw_lower for x in ["hydrogen tank", "pressure vessel", "气瓶", "压力容器"]):
        labels.append("app:pressure_vessel")
    if any(x in kw_lower for x in ["bicycle", "sports", "tennis", "helmet", "自行车", "运动"]):
        labels.append("app:sport")
    if any(x in kw_lower for x in ["prosthetic", "medical", "义肢", "假肢", "医疗"]):
        labels.append("app:medical")

    # 语言标签
    if any("\u4e00" <= ch <= "\u9fff" for ch in kw):
        labels.append("lang:zh")
    else:
        labels.append("lang:en")

    return labels


def print_progress(current, total, start_time):
    """
    简单进度条 + ETA。
    current: 当前完成的关键词数
    total: 总关键词数
    start_time: 脚本开始时间（time.time()）
    """
    elapsed = time.time() - start_time
    if current == 0:
        eta_str = "未知"
    else:
        avg = elapsed / current
        remaining = total - current
        eta = remaining * avg

        def fmt(sec):
            m = int(sec // 60)
            s = int(sec % 60)
            return f"{m:02d}:{s:02d}"

        eta_str = fmt(eta)

    progress = current / total if total else 0
    bar_len = 30
    filled = int(bar_len * progress)
    bar = "█" * filled + "-" * (bar_len - filled)
    pct = progress * 100

    def fmt(sec):
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m:02d}:{s:02d}"

    print(
        f"[{bar}] {pct:5.1f}% | 已用 {fmt(elapsed)} | 预估剩余 ~ {eta_str}",
        flush=True
    )


def merge_all_metadata(base_dir, output_path, dedupe_by="image_url"):
    """
    扫描 base_dir 下所有 *_metadata.json，合并为一个大文件。
    - dedupe_by: 按哪个字段去重（image_url 或 local_path）
    输出为 JSONL（每行一条记录），方便后续处理。
    """
    records = []
    seen = set()

    for root, dirs, files in os.walk(base_dir):
        for name in files:
            if not name.endswith(".json"):
                continue
            if "metadata" not in name:
                continue
            full_path = os.path.join(root, name)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"读取 {full_path} 失败：{e}")
                continue

            if not isinstance(data, list):
                continue

            for rec in data:
                key = rec.get(dedupe_by)
                if not key:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                records.append(rec)

    # 写 JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"合并完成：总记录数 {len(records)}，输出文件：{output_path}")


def preload_global_seen_urls(base_dir):
    """
    预加载历史 URL（可选功能）。
    默认在 run_batch 里不启用，以免把几乎所有结果都当成“旧图”。

    如果以后你希望“跨多次运行也不重复下载相同 image_url”，
    可以在 run_batch 里改成使用这个函数。
    """
    seen_urls = set()
    for root, dirs, files in os.walk(base_dir):
        for name in files:
            if not name.endswith(".json"):
                continue
            if "metadata" not in name:
                continue
            full_path = os.path.join(root, name)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"预加载 URL 时读取 {full_path} 失败：{e}")
                continue

            if not isinstance(data, list):
                continue

            for rec in data:
                url = rec.get("image_url")
                if url:
                    seen_urls.add(url)

    print(f"预加载全局 URL 去重集合完成，共 {len(seen_urls)} 个 URL。")
    return seen_urls


def run_batch(test_mode=True, start_idx=None, end_idx=None):
    """
    批量跑一段关键词区间。
    - test_mode=True 时只跑前几个关键词，调试用。
    - 可以通过 --start / --end 控制子区间，方便多机并行。
    """
    base_dir = "./imgs/carbon_fiber_mm"
    os.makedirs(base_dir, exist_ok=True)

    # 1. 获取全量关键词列表 & 每关键词的 max_count
    if test_mode:
        all_keywords = CARBON_FIBER_KEYWORDS[:5]
        per_keyword_max = 30
    else:
        all_keywords = CARBON_FIBER_KEYWORDS
        per_keyword_max = 120   # 可按需要调整上限

    total_all = len(all_keywords)

    # 2. 处理 start_idx / end_idx：限定本进程负责的子区间
    if start_idx is None:
        start_idx = 0
    if end_idx is None or end_idx > total_all:
        end_idx = total_all
    if start_idx < 0:
        start_idx = 0
    if start_idx >= end_idx:
        print(f"start_idx={start_idx} >= end_idx={end_idx}，没有任务可跑，直接退出。")
        return

    keywords = all_keywords[start_idx:end_idx]
    total = len(keywords)

    print(f"本进程将处理关键词区间 [{start_idx}, {end_idx})，共 {total} 个关键词。")

    batch_size = 35

    # 3. 全局 URL 去重集合：只在“本次进程”有效
    #    修法二：继续在本次进程内跨关键词去重，但不会导致后续关键词提前终止。
    global_seen_urls = set()

    # 如果以后你想恢复“读历史 URL 做跨多次运行去重”，可以改成：
    # global_seen_urls = preload_global_seen_urls(base_dir)

    start_time = time.time()

    # 4. 遍历当前分片内的所有关键词
    for offset, kw in enumerate(keywords):
        # 全局索引（基于 CARBON_FIBER_KEYWORDS 的 index）
        global_idx = start_idx + offset  # 0-based
        pretty_idx = global_idx + 1      # 人类友好的 1-based

        labels = label_keyword(kw)
        slug = slugify(kw)

        save_dir = os.path.join(base_dir, f"{pretty_idx:04d}_{slug}")
        metadata_filename = f"{slug}_metadata.json"
        file_prefix = f"{pretty_idx:04d}_"

        print(
            f"\n==== [全局 {pretty_idx}/{total_all}] "
            f"本进程 {offset+1}/{total} keyword: {kw} ====\n"
        )
        print("标签：", labels)

        try:
            crawl_bing_images(
                keyword=kw,
                save_dir=save_dir,
                max_count=per_keyword_max,
                batch_size=batch_size,
                file_prefix=file_prefix,
                metadata_filename=metadata_filename,
                global_seen_urls=global_seen_urls,   # 本次 run 内共享去重
                keyword_labels=labels,
                resume=True,                         # 有 metadata 就跳过
                max_workers=5
            )
        except Exception as e:
            print(f"关键词 '{kw}' 抓取时出错：{e}")

        # 进度条
        print_progress(offset + 1, total, start_time)

        # 每个关键词之间稍微停一下，避免太猛
        time.sleep(1)

    # 5. 本进程结束后：合并 metadata（输出子语料）
    merged_output = os.path.join(
        base_dir,
        f"carbon_fiber_corpus_{start_idx}_{end_idx}.jsonl"
    )
    merge_all_metadata(base_dir, merged_output, dedupe_by="image_url")
    print(f"本进程区间 [{start_idx}, {end_idx}) 合并完成：{merged_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="测试模式，只跑少量关键词")
    parser.add_argument("--start", type=int, default=None,
                        help="关键词起始下标（包含），默认从0开始")
    parser.add_argument("--end", type=int, default=None,
                        help="关键词结束下标（不包含），默认到总数")
    args = parser.parse_args()

    run_batch(
        test_mode=args.test,
        start_idx=args.start,
        end_idx=args.end
    )
