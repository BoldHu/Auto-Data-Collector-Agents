"""Image deduplicator — compute perceptual hashes, group near-duplicates, select primaries.

Phase 3.2: Deduplicate images using perceptual hashing (phash):
1. Load image_index.jsonl
2. Compute perceptual hash (imagehash phash) for each image
3. Group near-duplicates by hamming distance ≤ 8
4. Select best primary per group (largest file, highest metadata quality)
5. Mark duplicates with their primary reference
6. Write dedup results

Output:
    data/interim/image_dedup/image_dedup.jsonl
    data/reports/phase_3_image_labeling/phase_3_2_dedup_report.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import imagehash
from PIL import Image

from src.autodata.pipelines.image_schema import DedupStatus, SourceStatus
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("image_dedup")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INDEX_PATH = PROJECT_ROOT / "data" / "interim" / "image_index" / "image_index.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "image_dedup"
REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_3_image_labeling"


class ImageDeduplicator:
    """Compute perceptual hashes and deduplicate images."""

    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        output_dir: Path = OUTPUT_DIR,
        report_dir: Path = REPORT_DIR,
        hamming_threshold: int = 8,
        precomputed_dedup_path: Optional[Path] = None,
    ) -> None:
        self.index_path = index_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        self.hamming_threshold = hamming_threshold
        self.precomputed_dedup_path = precomputed_dedup_path
        self.start_time = time.time()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def load_index(self) -> list[dict]:
        """Load image index from JSONL."""
        logger.info(f"Loading index: {self.index_path}")
        items = []
        with open(self.index_path, encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        logger.info(f"Loaded {len(items)} index items")
        return items

    def load_precomputed_phashes(self, items: list[dict]) -> list[dict]:
        """Load precomputed phash values from a previous dedup run.

        Merges phash fields from the precomputed JSONL into current items,
        keyed by image_id.
        """
        if not self.precomputed_dedup_path or not self.precomputed_dedup_path.exists():
            return items

        logger.info(f"Loading precomputed phash values from {self.precomputed_dedup_path}")
        phash_lookup = {}
        with open(self.precomputed_dedup_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                phash_lookup[record["image_id"]] = record.get("phash", "")

        merged = 0
        missing = 0
        for item in items:
            if item["image_id"] in phash_lookup and phash_lookup[item["image_id"]]:
                item["phash"] = phash_lookup[item["image_id"]]
                merged += 1
            else:
                missing += 1

        logger.info(f"Precomputed phashes merged: {merged}, missing: {missing}")
        return items

    def compute_phashes(self, items: list[dict]) -> list[dict]:
        """Compute perceptual hash for each image. Add phash field."""
        logger.info(f"Computing perceptual hashes for {len(items)} images...")
        computed = 0
        failed = 0

        for item in items:
            try:
                img = Image.open(item["file_path"])
                phash = imagehash.phash(img)
                item["phash"] = str(phash)
                computed += 1
            except Exception as e:
                item["phash"] = ""
                failed += 1
                if failed <= 10:
                    logger.warning(f"phash failed for {item['file_path']}: {str(e)[:60]}")

        logger.info(f"phash computed: {computed}, failed: {failed}")
        return items

    def group_duplicates(self, items: list[dict]) -> dict[str, list[list[dict]]]:
        """Group near-duplicates by perceptual hash similarity.

        Uses a Union-Find approach:
        1. Exact phash matches → same group (definite duplicates)
        2. Near-duplicates (hamming distance ≤ threshold) → merge groups
        3. Each item appears in exactly one group

        Returns dict with:
            "unique": list of items with no duplicates (solo groups)
            "duplicate_groups": list of groups with ≥2 members
        """
        logger.info("Grouping near-duplicates...")

        hashed_items = [i for i in items if i["phash"]]
        no_hash_items = [i for i in items if not i["phash"]]

        # Union-Find: map each image_id to a group representative
        parent = {}  # image_id -> representative image_id

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        # Initialize: each item is its own group
        for item in hashed_items:
            parent[item["image_id"]] = item["image_id"]

        # Step 1: Merge exact phash matches
        phash_groups = {}
        for item in hashed_items:
            ph = item["phash"]
            if ph not in phash_groups:
                phash_groups[ph] = []
            phash_groups[ph].append(item)

        for ph, group in phash_groups.items():
            if len(group) > 1:
                rep = group[0]["image_id"]
                for item in group[1:]:
                    union(item["image_id"], rep)

        # Step 2: Merge near-duplicates across different phashes
        unique_phashes = list(phash_groups.keys())
        logger.info(f"Comparing {len(unique_phashes)} unique phash values for near-duplicates...")

        for i in range(len(unique_phashes)):
            hash_i = imagehash.hex_to_hash(unique_phashes[i])
            for j in range(i + 1, len(unique_phashes)):
                hash_j = imagehash.hex_to_hash(unique_phashes[j])
                if hash_i - hash_j <= self.hamming_threshold:
                    # Merge groups: connect first item of each phash group
                    rep_i = phash_groups[unique_phashes[i]][0]["image_id"]
                    rep_j = phash_groups[unique_phashes[j]][0]["image_id"]
                    union(rep_i, rep_j)

        # Step 3: Build groups from union-find results
        groups_by_rep = {}
        for item in hashed_items:
            rep = find(item["image_id"])
            if rep not in groups_by_rep:
                groups_by_rep[rep] = []
            groups_by_rep[rep].append(item)

        # Categorize: solo groups are "unique", multi-member groups are "duplicate_groups"
        duplicate_groups = []
        unique_items = []
        for rep, group in groups_by_rep.items():
            if len(group) >= 2:
                duplicate_groups.append(group)
            else:
                unique_items.append(group[0])

        # Items without phash are treated as unique (unknown)
        for item in no_hash_items:
            unique_items.append(item)

        logger.info(
            f"Duplicate groups: {len(duplicate_groups)}, "
            f"Unique: {len(unique_items)}, "
            f"No-hash: {len(no_hash_items)}, "
            f"Total: {len(unique_items) + sum(len(g) for g in duplicate_groups)}"
        )

        return {
            "unique": unique_items,
            "duplicate_groups": duplicate_groups,
        }

    def select_primaries(self, groups: dict) -> list[dict]:
        """Select best primary per duplicate group and mark others.

        Selection criteria for primary:
        1. Has metadata (METADATA_MATCHED or PATH_REPAIRED) preferred over METADATA_MISSING
        2. Largest file size (higher resolution)
        3. Has image_url (more metadata completeness)

        All non-primary items get dedup_status=DUPLICATE and reference their primary.
        Primary items get dedup_status=UNIQUE.
        Unique items (no group) get dedup_status=UNIQUE.
        """
        logger.info("Selecting primaries...")

        result_items = []

        # Process unique items
        for item in groups["unique"]:
            item["dedup_status"] = DedupStatus.UNIQUE.value
            item["dedup_primary_id"] = ""
            item["dedup_group_id"] = ""
            result_items.append(item)

        # Process duplicate groups
        total_duplicate_groups = len(groups["duplicate_groups"])
        total_duplicates = 0

        for group_idx, group in enumerate(groups["duplicate_groups"]):
            group_id = f"dedup_grp_{group_idx}"

            # Select primary: prefer metadata > size > has_url
            def primary_score(item):
                has_metadata = item.get("source_status", "") in (
                    SourceStatus.METADATA_MATCHED.value,
                    SourceStatus.PATH_REPAIRED.value,
                )
                has_url = bool(item.get("image_url", ""))
                file_size = item.get("file_size", 0)
                return (has_metadata, has_url, file_size)

            sorted_group = sorted(group, key=primary_score, reverse=True)
            primary = sorted_group[0]

            primary["dedup_status"] = DedupStatus.UNIQUE.value
            primary["dedup_primary_id"] = primary["image_id"]
            primary["dedup_group_id"] = group_id
            result_items.append(primary)

            # Mark duplicates
            for dup in sorted_group[1:]:
                dup["dedup_status"] = DedupStatus.DUPLICATE.value
                dup["dedup_primary_id"] = primary["image_id"]
                dup["dedup_group_id"] = group_id
                total_duplicates += 1
                result_items.append(dup)

        logger.info(
            f"Selected primaries: {len(groups['duplicate_groups'])}, "
            f"Total duplicates: {total_duplicates}, "
            f"Total unique: {sum(1 for i in result_items if i['dedup_status'] == 'unique')}"
        )

        return result_items

    def write_dedup_results(self, items: list[dict]) -> Path:
        """Write dedup results to JSONL."""
        output_path = self.output_dir / "image_dedup.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Written {len(items)} items to {output_path}")
        return output_path

    def write_report(self, items: list[dict], groups: dict) -> Path:
        """Write Phase 3.2 dedup report."""
        unique_count = sum(1 for i in items if i["dedup_status"] == DedupStatus.UNIQUE.value)
        duplicate_count = sum(1 for i in items if i["dedup_status"] == DedupStatus.DUPLICATE.value)
        near_dup_count = sum(1 for i in items if i["dedup_status"] == DedupStatus.NEAR_DUPLICATE.value)
        unknown_count = sum(1 for i in items if i["dedup_status"] == DedupStatus.UNKNOWN.value)

        # Dedup group stats
        group_sizes = [len(g) for g in groups["duplicate_groups"]]
        group_size_dist = {}
        for s in group_sizes:
            group_size_dist[s] = group_size_dist.get(s, 0) + 1

        # Effective unique corpus after dedup
        effective_unique = unique_count

        report = {
            "phase": "3.2",
            "run_id": f"phase_3_2_{int(self.start_time)}",
            "timestamp": time.time(),
            "total_images_input": len(items),
            "total_unique": unique_count,
            "total_duplicates": duplicate_count,
            "total_near_duplicates": near_dup_count,
            "total_unknown_dedup": unknown_count,
            "total_duplicate_groups": len(groups["duplicate_groups"]),
            "effective_unique_after_dedup": effective_unique,
            "dedup_rate": duplicate_count / len(items) if items else 0,
            "group_size_distribution": group_size_dist,
            "hamming_threshold": self.hamming_threshold,
            "errors": [],
        }

        report_path = self.report_dir / "phase_3_2_dedup_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Written report to {report_path}")
        return report_path

    def run(self) -> dict[str, Any]:
        """Execute the complete dedup pipeline."""
        logger.info("=== Phase 3.2: Image Deduplication ===")

        # Step 1: Load index
        items = self.load_index()

        # Step 2: Load precomputed phash values if available, otherwise compute
        if self.precomputed_dedup_path and self.precomputed_dedup_path.exists():
            items = self.load_precomputed_phashes(items)
            # Compute phashes for items that didn't have them
            items_without_phash = [i for i in items if not i.get("phash")]
            if items_without_phash:
                logger.info(f"Computing phash for {len(items_without_phash)} items without precomputed values")
                self.compute_phashes(items_without_phash)
        else:
            items = self.compute_phashes(items)

        # Step 3: Group duplicates
        groups = self.group_duplicates(items)

        # Step 4: Select primaries and mark duplicates
        result_items = self.select_primaries(groups)

        # Step 5: Write dedup results
        dedup_path = self.write_dedup_results(result_items)

        # Step 6: Write report
        report_path = self.write_report(result_items, groups)

        elapsed = time.time() - self.start_time
        logger.info(f"=== Phase 3.2 complete in {elapsed:.1f}s ===")

        return {
            "dedup_path": str(dedup_path),
            "report_path": str(report_path),
            "total_indexed": len(result_items),
            "elapsed_seconds": elapsed,
        }