from __future__ import annotations

import logging
from pathlib import Path

import fitz
import torch
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from RAG_Agent.infrastructure.ingestion.exceptions import PDFParsingException, PDFValidationError
from RAG_Agent.infrastructure.ingestion.ingest_profile import LOCAL

logger = logging.getLogger(__name__)


class DoclingExtractor:
    """Extrae un DoclingDocument desde un PDF (texto nativo u OCR previo).

    ``max_pages`` is the per-call ceiling (typically ``pages_per_shard``). Long
    documents are split by ``NativePdfPipeline`` via ``page_range``; this class
    does not reject a file solely because ``page_count`` is large when a range
    is provided.

    Args:
        layout_batch_size: Pages per layout-model forward pass.
        table_batch_size: Tables per TableFormer forward pass.
        table_former_mode: TableFormer quality. Pin ``ACCURATE`` so version
            upgrades cannot silently switch.
    """

    def __init__(
        self,
        max_pages: int = LOCAL.pages_per_shard,
        max_file_size_mb: int = LOCAL.max_file_size_mb,
        do_ocr: bool = False,
        do_table_structure: bool = True,
        layout_batch_size: int = LOCAL.layout_batch_size,
        table_batch_size: int = LOCAL.table_batch_size,
        table_former_mode: TableFormerMode | str = LOCAL.table_former_mode,
    ) -> None:
        pipeline_options = PdfPipelineOptions(
            do_table_structure=do_table_structure,
            do_ocr=do_ocr,
            layout_batch_size=layout_batch_size,
            table_batch_size=table_batch_size,
        )
        pipeline_options.table_structure_options.mode = TableFormerMode(table_former_mode)

        if torch.cuda.is_available():
            logger.info("GPU detected: %s", torch.cuda.get_device_name(0))
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=4,
                device=AcceleratorDevice.CUDA,
            )
        else:
            logger.warning("Running Docling on CPU; conversion may be slow")

        self._pipeline_options = pipeline_options
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)},
        )
        self._max_pages = max_pages
        self._max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def extract(
        self,
        pdf_path: Path,
        *,
        page_range: tuple[int, int] | None = None,
    ) -> DoclingDocument:
        """Convert one PDF, optionally a 1-based inclusive page range.

        Args:
            pdf_path: Cleaned PDF (full document, never a sub-PDF).
            page_range: Inclusive ``(first_page, last_page)`` in PDF folio numbers.
                ``None`` only if ``page_count <= max_pages``.

        Returns:
            DoclingDocument for that range (or the whole file).
        """
        pdf_path = Path(pdf_path)
        page_count = self._validate_pdf(pdf_path, page_range=page_range)
        if page_range is None:
            logger.info("Extracting PDF (%d pages): %s", page_count, pdf_path.name)
        else:
            lo, hi = page_range
            logger.info(
                "Extracting PDF pages %d-%d of %d: %s",
                lo,
                hi,
                page_count,
                pdf_path.name,
            )
        return self._convert_file(pdf_path, page_range=page_range, page_count=page_count)

    def _convert_file(
        self,
        pdf_path: Path,
        *,
        page_range: tuple[int, int] | None = None,
        page_count: int,
    ) -> DoclingDocument:
        # Docling compares max_num_pages to the full file's page_count, not the
        # range length. Range size is already enforced in _validate_pdf.
        try:
            if page_range is None:
                result = self._converter.convert(
                    str(pdf_path),
                    max_num_pages=self._max_pages,
                    max_file_size=self._max_file_size_bytes,
                )
            else:
                result = self._converter.convert(
                    str(pdf_path),
                    page_range=page_range,
                    max_num_pages=page_count,
                    max_file_size=self._max_file_size_bytes,
                )
        except Exception as exc:
            msg = f"Docling failed to convert {pdf_path.name}: {exc}"
            raise PDFParsingException(msg) from exc

        if result.document is None:
            msg = f"Docling returned no document for {pdf_path.name}"
            raise PDFParsingException(msg)

        return result.document

    def _validate_pdf(
        self,
        pdf_path: Path,
        page_range: tuple[int, int] | None = None,
    ) -> int:
        if not pdf_path.exists():
            raise PDFValidationError(f"PDF not found: {pdf_path}")

        file_size = pdf_path.stat().st_size
        if file_size == 0:
            raise PDFValidationError(f"PDF file is empty: {pdf_path}")

        if file_size > self._max_file_size_bytes:
            raise PDFValidationError(
                f"PDF too large: {file_size / 1024 / 1024:.1f}MB > "
                f"{self._max_file_size_bytes / 1024 / 1024:.1f}MB"
            )

        with pdf_path.open("rb") as pdf_file:
            if not pdf_file.read(8).startswith(b"%PDF-"):
                raise PDFValidationError(f"File does not have PDF header: {pdf_path}")

        page_count = count_pdf_pages(pdf_path)
        if page_range is None:
            if page_count > self._max_pages:
                raise PDFValidationError(
                    f"PDF shard/page batch too large: {page_count} > {self._max_pages}. "
                    "NativePdfPipeline should split before extract."
                )
            return page_count

        lo, hi = page_range
        if not (1 <= lo <= hi <= page_count):
            raise PDFValidationError(
                f"Invalid page_range {(lo, hi)} for PDF with {page_count} pages"
            )
        range_pages = hi - lo + 1
        if range_pages > self._max_pages:
            raise PDFValidationError(
                f"page_range {(lo, hi)} has {range_pages} pages > {self._max_pages}"
            )
        return page_count


def count_pdf_pages(pdf_path: Path) -> int:
    """Cuenta páginas con PyMuPDF (mismo backend que preprocess)."""
    with fitz.open(pdf_path) as doc:
        return doc.page_count
