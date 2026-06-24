"""Source split guard for leakage prevention.

Ensures same source documents stay in one split and
prevents cross-split contamination.
"""

from __future__ import annotations

from collections import defaultdict


class SourceSplitGuard:
    """Guards against source-level leakage in train/val splits."""

    def __init__(self):
        self.source_to_split: dict[str, str] = {}
        self.group_to_split: dict[str, str] = {}

    def assign_split(
        self,
        sample: dict,
        preferred_split: str = "train",
    ) -> str:
        """Assign a sample to a split, respecting source constraints.

        Rules:
        1. Same leakage_group_id must stay in same split
        2. Same source_file should prefer same split
        """
        leakage_group = sample.get("leakage_group_id", "")
        source_refs = sample.get("source_refs", [])

        # Check leakage group constraint
        if leakage_group and leakage_group in self.group_to_split:
            return self.group_to_split[leakage_group]

        # Check source ref constraint
        for ref in source_refs:
            if ref in self.source_to_split:
                split = self.source_to_split[ref]
                if leakage_group:
                    self.group_to_split[leakage_group] = split
                return split

        # No constraint found, use preferred split
        if leakage_group:
            self.group_to_split[leakage_group] = preferred_split
        for ref in source_refs:
            self.source_to_split[ref] = preferred_split

        return preferred_split

    def validate_splits(
        self,
        train: list[dict],
        val: list[dict],
    ) -> dict:
        """Validate that splits don't have source contamination.

        Returns:
            dict with validation results.
        """
        train_groups = set()
        val_groups = set()
        train_sources = set()
        val_sources = set()

        for s in train:
            lg = s.get("leakage_group_id", "")
            if lg:
                train_groups.add(lg)
            for ref in s.get("source_refs", []):
                train_sources.add(ref)

        for s in val:
            lg = s.get("leakage_group_id", "")
            if lg:
                val_groups.add(lg)
            for ref in s.get("source_refs", []):
                val_sources.add(ref)

        group_overlap = train_groups & val_groups
        source_overlap = train_sources & val_sources

        return {
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "group_overlap": len(group_overlap),
            "source_overlap": len(source_overlap),
            "overlapping_groups": list(group_overlap)[:10],
            "overlapping_sources": list(source_overlap)[:10],
            "valid": len(group_overlap) == 0,
        }
