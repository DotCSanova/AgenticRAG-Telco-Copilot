from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from RAG_Agent.domain.doc_processing_rules.document_processing import DocumentProfileResolver
from RAG_Agent.domain.value_objects.canonical_document import CanonicalDocument
from RAG_Agent.domain.value_objects.canonical_merge import merge_canonical_shards
from RAG_Agent.infrastructure.ingestion.exceptions import PDFParsingException
from RAG_Agent.infrastructure.ingestion.extractors.docling_extractor import (
    DoclingExtractor,
    count_pdf_pages,
)
from RAG_Agent.infrastructure.ingestion.ingest_profile import (
    IngestHardwareProfile,
    get_ingest_profile,
)
from RAG_Agent.infrastructure.ingestion.normalizers.docling_normalizer import DoclingNormalizer
from RAG_Agent.infrastructure.ingestion.preprocessing.pymupdf_preprocessor import (
    PyMuPDFPreprocessor,
)

logger = logging.getLogger(__name__)


def _page_offset_for_range(lo: int) -> int:
    """Merge offset for Docling ``page_range`` starting at 1-based ``lo``.

    Docling keeps original folio numbers (``page_no = i + 1`` on the full file),
    so this is 0. A backend that renormalized the slice to 1..N would return
    ``lo - 1``.
    """
    return 0


class NativePdfPipeline:
    """Pipeline PDF nativo: preprocesado → Docling (por page_range) → CanonicalDocument.

    Implementa el puerto DocumentParser. El perfil documental se inyecta vía
    DocumentProfileResolver. El perfil de hardware controla el tamaño de shard.
    """

    def __init__(
        self,
        profile_resolver: DocumentProfileResolver,
        extractor: DoclingExtractor | None = None,
        normalizer: DoclingNormalizer | None = None,
        hardware: IngestHardwareProfile | None = None,
    ) -> None:
        self._resolver = profile_resolver
        self._hardware = hardware or get_ingest_profile()
        self._extractor = extractor or DoclingExtractor(
            max_pages=self._hardware.pages_per_shard,
            max_file_size_mb=self._hardware.max_file_size_mb,
            do_ocr=False,
        )
        self._normalizer = normalizer or DoclingNormalizer()

    def parse(self, path: Path) -> CanonicalDocument:
        path = Path(path)
        profile = self._resolver.resolve(path)
        hw = self._hardware
        shard_size = hw.pages_per_shard

        with tempfile.TemporaryDirectory(prefix="rag_pipeline_") as tmp_dir:
            tmp = Path(tmp_dir)
            cleaned_path = tmp / f"{path.stem}_cleaned.pdf"
            PyMuPDFPreprocessor(profile.rules).preprocess(path, cleaned_path)
            logger.info("Preprocessed PDF ready for Docling: %s", cleaned_path.name)

            page_count = count_pdf_pages(cleaned_path)
            logger.info(
                "Ingest profile=%s pages=%d shard_size=%d file=%s",
                hw.name,
                page_count,
                shard_size,
                cleaned_path.name,
            )

            if page_count <= shard_size:
                docling_document = self._extractor.extract(cleaned_path)
                return self._normalizer.normalize(
                    docling_document,
                    source_path=path,
                    profile=profile,
                    parser_name="native_pdf_docling",
                )

            page_ranges = [
                (lo, min(lo + shard_size - 1, page_count))
                for lo in range(1, page_count + 1, shard_size)
            ]
            logger.info("Split into %d page ranges for Docling", len(page_ranges))

            parts: list[tuple[int, CanonicalDocument]] = []
            failed_shards: list[int] = []
            for index, (lo, hi) in enumerate(page_ranges):
                logger.info(
                    "Docling shard %d/%d (pages %d-%d)",
                    index + 1,
                    len(page_ranges),
                    lo,
                    hi,
                )
                try:
                    docling_document = self._extractor.extract(
                        cleaned_path, page_range=(lo, hi)
                    )
                    canonical = self._normalizer.normalize(
                        docling_document,
                        source_path=path,
                        profile=profile,
                        parser_name="native_pdf_docling",
                    )
                except PDFParsingException:
                    logger.error("Shard %d failed (pages %d-%d), skipping", index, lo, hi)
                    failed_shards.append(index)
                    continue
                parts.append((_page_offset_for_range(lo), canonical))

            if not parts:
                raise PDFParsingException(
                    f"All {len(page_ranges)} shards failed for {path.name}"
                )

            extra = dict(profile.identity.metadata)
            extra["ingest_profile"] = hw.name
            extra["pages_per_shard"] = str(shard_size)
            if failed_shards:
                extra["failed_shards"] = ",".join(map(str, failed_shards))

            return merge_canonical_shards(
                parts,
                source_path=path,
                profile_id=profile.rules.profile_id,
                parser_name="native_pdf_docling",
                rules=profile.rules,
                build_sections=lambda blocks: self._normalizer.build_sections(
                    blocks, rules=profile.rules
                ),
                resolve_title=lambda sections, blocks, hint: self._normalizer.resolve_title(
                    sections, blocks, hint, rules=profile.rules
                ),
                title_hint=profile.identity.title_hint,
                extra=extra,
            )
