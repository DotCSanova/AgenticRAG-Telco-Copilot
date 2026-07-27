from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import fitz

from RAG_Agent.domain.doc_processing_rules.default_document_rules import DEFAULT_DOCUMENT_RULES
from RAG_Agent.domain.doc_processing_rules.document_processing import (
    DocumentProcessingRules,
    PageLayoutThresholds,
    PreprocessOptions,
)

logger = logging.getLogger(__name__)


class PyMuPDFPreprocessor:
    """Limpia PDFs sin eliminar páginas: chrome (headers/footers/logos) y portada.

    El preprocessor es agnóstico al tipo de documento. Qué limpia lo decide
    ``rules.preprocess_options``. El front matter semántico lo filtra el
    normalizer. La numeración de páginas se conserva.
    """

    def __init__(self, rules: DocumentProcessingRules | None = None) -> None:
        self._rules = rules or DEFAULT_DOCUMENT_RULES
        self._layout: PageLayoutThresholds = self._rules.page_layout
        self._options: PreprocessOptions = self._rules.preprocess_options

    def preprocess(self, input_path: Path | str, output_path: Path | str) -> None:
        """Analiza el PDF original y guarda una copia limpia en ``output_path``."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        options = self._options

        noisy_texts, logo_xrefs = self._analyze(input_path)

        with fitz.open(input_path) as doc:
            original_page_count = len(doc)

            if options.clean_repeated_headers_footers:
                self._apply_header_footer_text_cleanup(doc, noisy_texts)

            if options.clean_cover_page and len(doc) > 0:
                self._clean_first_page(doc)

            if options.clean_header_footer_images:
                self._remove_header_footer_images(doc, logo_xrefs)

            if len(doc) != original_page_count:
                msg = (
                    f"El preprocessor no debe eliminar páginas: "
                    f"antes={original_page_count}, después={len(doc)}"
                )
                raise RuntimeError(msg)

            doc.save(output_path)

        logger.info("PDF procesado guardado en: %s (%d páginas)", output_path, original_page_count)

    def _analyze(self, input_path: Path) -> tuple[set[tuple[str, str]], set[int]]:
        with fitz.open(input_path) as doc:
            pages_blocks = self._extract_blocks(doc)
            noisy_texts = self._detect_repeated_text_elements(pages_blocks)
            logo_xrefs = self._detect_repeated_logo_xrefs(doc)
        return noisy_texts, logo_xrefs

    def _block_zone(self, y0: float, y1: float, page_height: float) -> str:
        if y0 < self._layout.header_top_ratio * page_height:
            return "top"
        if y1 > self._layout.footer_bottom_ratio * page_height:
            return "bottom"
        return "body"

    def _extract_blocks(self, doc: fitz.Document) -> list[list[dict[str, float | str]]]:
        pages_blocks: list[list[dict[str, float | str]]] = []

        for page in doc:
            page_height = page.rect.height
            page_data: list[dict[str, float | str]] = []

            for block in page.get_text("blocks"):
                _, y0, _, y1, text, *_ = block
                page_data.append(
                    {
                        "text": text.strip(),
                        "y0": y0,
                        "y1": y1,
                        "height": page_height,
                    }
                )

            pages_blocks.append(page_data)

        return pages_blocks

    def _detect_repeated_text_elements(
        self,
        pages_blocks: list[list[dict[str, float | str]]],
    ) -> set[tuple[str, str]]:
        counter: Counter[tuple[str, str]] = Counter()
        total_pages = len(pages_blocks)

        if total_pages == 0:
            return set()

        for page in pages_blocks:
            seen: set[tuple[str, str]] = set()
            for block in page:
                text = self._rules.normalize_text(str(block["text"]))
                if not text:
                    continue

                zone = self._block_zone(float(block["y0"]), float(block["y1"]), float(block["height"]))
                signature = (text, zone)
                if signature not in seen:
                    counter[signature] += 1
                    seen.add(signature)

        min_ratio = self._rules.repeated_element_min_ratio
        return {
            signature
            for signature, count in counter.items()
            if count / total_pages >= min_ratio
        }

    def _detect_repeated_logo_xrefs(self, doc: fitz.Document) -> set[int]:
        image_counter: Counter[int] = Counter()
        for page in doc:
            for image in page.get_images(full=True):
                image_counter[image[0]] += 1

        total_pages = len(doc)
        if total_pages == 0:
            return set()

        min_ratio = self._rules.repeated_element_min_ratio
        return {
            xref
            for xref, count in image_counter.items()
            if count / total_pages >= min_ratio
        }

    def _apply_header_footer_text_cleanup(
        self,
        doc: fitz.Document,
        noisy_texts: set[tuple[str, str]],
    ) -> None:
        for page in doc:
            page_height = page.rect.height
            for block in page.get_text("blocks"):
                x0, y0, x1, y1, text, *_ = block
                normalized = self._rules.normalize_text(text)
                zone = self._block_zone(y0, y1, page_height)
                if (normalized, zone) in noisy_texts:
                    page.add_redact_annot(fitz.Rect(x0, y0, x1, y1))
            page.apply_redactions()

    def _remove_header_footer_images(self, doc: fitz.Document, logo_xrefs: set[int]) -> None:
        """Redacta logos repetidos y cualquier imagen en zonas de header/footer."""
        for page in doc:
            page_height = page.rect.height

            for image in page.get_images(full=True):
                xref = image[0]
                for rect in page.get_image_rects(xref):
                    center_y = (rect.y0 + rect.y1) / 2
                    in_header_footer = (
                        center_y < self._layout.header_top_ratio * page_height
                        or center_y > self._layout.footer_bottom_ratio * page_height
                    )
                    if xref in logo_xrefs or in_header_footer:
                        page.add_redact_annot(rect)

            page.apply_redactions()

    def _clean_first_page(self, doc: fitz.Document) -> None:
        page = doc[0]
        page_width = page.rect.width
        page_height = page.rect.height

        block_font_sizes: dict[tuple[int, ...], float] = {}
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue

            bbox_key = tuple(round(coord) for coord in block["bbox"])
            max_size = 0.0
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    max_size = max(max_size, span.get("size", 0))
            block_font_sizes[bbox_key] = max_size

        blocks = page.get_text("blocks")
        title_block = self._find_title_block(blocks, page_width, page_height, block_font_sizes)

        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            if title_block is not None and (x0, y0, x1, y1) == tuple(title_block[:4]):
                continue
            page.add_redact_annot(fitz.Rect(x0, y0, x1, y1))

        for image in page.get_images(full=True):
            for rect in page.get_image_rects(image[0]):
                page.add_redact_annot(rect)

        page.apply_redactions()

    def _find_title_block(
        self,
        blocks: list[tuple],
        page_width: float,
        page_height: float,
        block_font_sizes: dict[tuple[int, ...], float],
    ) -> tuple | None:
        candidates: list[tuple[float, float, tuple]] = []
        title_top = self._layout.title_zone_top
        title_bottom = self._layout.title_zone_bottom

        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            text = text.strip()
            if not text:
                continue
            if y0 < title_top * page_height or y1 > title_bottom * page_height:
                continue

            center_x = (x0 + x1) / 2.0
            block_width = max(x1 - x0, 1.0)
            centering_score = abs(center_x - page_width / 2.0) / block_width
            bbox_key = (round(x0), round(y0), round(x1), round(y1))
            font_size = block_font_sizes.get(bbox_key, 0.0)
            candidates.append((centering_score, -font_size, block))

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]))
            return candidates[0][2]

        for block in blocks:
            x0, y0, _, _, text, *_ = block
            if text.strip() and title_top * page_height < y0 < title_bottom * page_height:
                return block

        return None
