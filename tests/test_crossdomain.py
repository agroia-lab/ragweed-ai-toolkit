"""
Tests for ragweed_toolkit.crossdomain module.

Covers validate_labels format detection and process_yolo_labels parsing.
"""

import tempfile
from pathlib import Path

import pytest

from ragweed_toolkit.crossdomain.dataset_builder import (
    process_yolo_labels,
    validate_labels,
)


# ------------------------------------------------------------------ #
# validate_labels
# ------------------------------------------------------------------ #

class TestValidateLabels:
    """Test validate_labels returns correct format issues."""

    def test_valid_labels_no_issues(self, tmp_path):
        """Correctly formatted YOLO labels should produce no issues."""
        label_file = tmp_path / "img001.txt"
        label_file.write_text(
            "0 0.500000 0.500000 0.100000 0.100000\n"
            "0 0.250000 0.750000 0.050000 0.080000\n"
        )
        issues = validate_labels(tmp_path, n=5)
        assert issues == []

    def test_wrong_class_id(self, tmp_path):
        """Class != 0 should be flagged."""
        label_file = tmp_path / "img002.txt"
        label_file.write_text("3 0.5 0.5 0.1 0.1\n")
        issues = validate_labels(tmp_path, n=5)
        assert any("class=3" in issue for issue in issues)

    def test_wrong_field_count(self, tmp_path):
        """Lines with != 5 fields should be flagged."""
        label_file = tmp_path / "img003.txt"
        label_file.write_text("0 0.5 0.5\n")  # only 3 fields
        issues = validate_labels(tmp_path, n=5)
        assert any("expected 5 fields" in issue for issue in issues)

    def test_coord_out_of_range(self, tmp_path):
        """Coordinates > 1 should be flagged."""
        label_file = tmp_path / "img004.txt"
        label_file.write_text("0 1.5 0.5 0.1 0.1\n")
        issues = validate_labels(tmp_path, n=5)
        assert any("out of [0,1]" in issue for issue in issues)

    def test_negative_coord(self, tmp_path):
        """Negative coordinates should be flagged."""
        label_file = tmp_path / "img005.txt"
        label_file.write_text("0 -0.1 0.5 0.1 0.1\n")
        issues = validate_labels(tmp_path, n=5)
        assert any("out of [0,1]" in issue for issue in issues)

    def test_empty_directory(self, tmp_path):
        """Empty directory should report no label files."""
        issues = validate_labels(tmp_path, n=5)
        assert issues == ["No label files found"]

    def test_multiple_files(self, tmp_path):
        """Multiple valid files should produce no issues."""
        for i in range(5):
            (tmp_path / f"img{i:03d}.txt").write_text(
                f"0 {0.1 * (i + 1):.6f} 0.500000 0.100000 0.100000\n"
            )
        issues = validate_labels(tmp_path, n=5)
        assert issues == []


# ------------------------------------------------------------------ #
# process_yolo_labels
# ------------------------------------------------------------------ #

class TestProcessYoloLabels:
    """Test process_yolo_labels parses standard YOLO format."""

    def test_single_class_filter(self, tmp_path):
        """Filter ragweed class 0 from single-class file."""
        label_file = tmp_path / "test.txt"
        label_file.write_text(
            "0 0.500000 0.500000 0.100000 0.100000\n"
            "0 0.250000 0.750000 0.050000 0.080000\n"
        )
        result = process_yolo_labels(label_file, ragweed_class=0, filter_ragweed_only=True)
        assert result is not None
        assert len(result) == 2
        # All lines should be remapped to class 0
        for line in result:
            assert line.startswith("0 ")

    def test_multi_class_filter(self, tmp_path):
        """Filter only ragweed class from multi-class file."""
        label_file = tmp_path / "test.txt"
        label_file.write_text(
            "0 0.5 0.5 0.1 0.1\n"
            "1 0.3 0.3 0.2 0.2\n"
            "2 0.7 0.7 0.05 0.05\n"
        )
        result = process_yolo_labels(label_file, ragweed_class=1, filter_ragweed_only=True)
        assert result is not None
        assert len(result) == 1
        assert result[0].startswith("0 ")

    def test_no_matching_class_returns_none(self, tmp_path):
        """If no lines match ragweed class, return None."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("1 0.5 0.5 0.1 0.1\n")
        result = process_yolo_labels(label_file, ragweed_class=99, filter_ragweed_only=True)
        assert result is None

    def test_nonexistent_file_returns_none(self, tmp_path):
        """Missing file should return None."""
        result = process_yolo_labels(
            tmp_path / "nonexistent.txt", ragweed_class=0, filter_ragweed_only=True
        )
        assert result is None

    def test_preserves_bbox_coords(self, tmp_path):
        """Bounding box coordinates should be preserved."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.123456 0.654321 0.100000 0.200000\n")
        result = process_yolo_labels(label_file, ragweed_class=0, filter_ragweed_only=True)
        assert result is not None
        parts = result[0].split()
        assert parts[0] == "0"
        assert parts[1] == "0.123456"
        assert parts[2] == "0.654321"
        assert parts[3] == "0.100000"
        assert parts[4] == "0.200000"

    def test_skips_short_lines(self, tmp_path):
        """Lines with fewer than 5 parts should be skipped."""
        label_file = tmp_path / "test.txt"
        label_file.write_text(
            "0 0.5 0.5\n"   # too short
            "0 0.5 0.5 0.1 0.1\n"  # valid
        )
        result = process_yolo_labels(label_file, ragweed_class=0, filter_ragweed_only=True)
        assert result is not None
        assert len(result) == 1
