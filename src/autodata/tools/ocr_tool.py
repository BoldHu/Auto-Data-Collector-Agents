"""OCR tool for scanned documents and images.

Wraps Tesseract OCR with Chinese + English support.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def ocr_image(
    image_path: Path,
    languages: str = "chi_sim+eng",
    layout: bool = True,
) -> dict:
    """Run OCR on an image file.

    Args:
        image_path: Path to image file
        languages: Tesseract language codes
        layout: Whether to preserve layout

    Returns:
        Dict with 'text', 'success', 'method', 'error'
    """
    try:
        cmd = ["tesseract", str(image_path), "stdout", "-l", languages]
        if layout:
            cmd.extend(["--psm", "6"])  # Assume uniform block of text

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            return {
                "text": result.stdout.strip(),
                "success": True,
                "method": "tesseract",
                "error": None,
            }
        else:
            return {
                "text": "",
                "success": False,
                "method": "tesseract",
                "error": result.stderr[:200],
            }

    except FileNotFoundError:
        return {
            "text": "",
            "success": False,
            "method": "tesseract",
            "error": "Tesseract not installed",
        }
    except subprocess.TimeoutExpired:
        return {
            "text": "",
            "success": False,
            "method": "tesseract",
            "error": "OCR timeout",
        }
    except Exception as e:
        return {
            "text": "",
            "success": False,
            "method": "tesseract",
            "error": str(e)[:200],
        }


def ocr_pdf_pages(
    pdf_path: Path,
    languages: str = "chi_sim+eng",
    max_pages: int = 50,
) -> list[dict]:
    """Run OCR on each page of a PDF.

    Converts PDF to images first, then OCR each page.

    Args:
        pdf_path: Path to PDF file
        languages: Tesseract language codes
        max_pages: Maximum pages to process

    Returns:
        List of dicts with 'page_number', 'text', 'success', 'error'
    """
    results = []

    try:
        # Convert PDF to images using pdftoppm
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["pdftoppm", "-png", "-r", "200", str(pdf_path),
                 str(Path(tmpdir) / "page")],
                capture_output=True, text=True, timeout=300,
            )

            if result.returncode != 0:
                return [{"page_number": 0, "text": "", "success": False,
                         "error": f"pdftoppm failed: {result.stderr[:200]}"}]

            # Find generated images
            page_images = sorted(Path(tmpdir).glob("page-*.png"))

            for i, img_path in enumerate(page_images[:max_pages]):
                ocr_result = ocr_image(img_path, languages)
                results.append({
                    "page_number": i + 1,
                    "text": ocr_result["text"],
                    "success": ocr_result["success"],
                    "error": ocr_result["error"],
                })

    except FileNotFoundError:
        results.append({
            "page_number": 0,
            "text": "",
            "success": False,
            "error": "pdftoppm not installed (poppler-utils)",
        })
    except subprocess.TimeoutExpired:
        results.append({
            "page_number": 0,
            "text": "",
            "success": False,
            "error": "PDF to image conversion timeout",
        })
    except Exception as e:
        results.append({
            "page_number": 0,
            "text": "",
            "success": False,
            "error": str(e)[:200],
        })

    return results


def is_tesseract_available() -> bool:
    """Check if Tesseract is installed."""
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
