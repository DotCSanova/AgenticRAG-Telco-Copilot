from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import fitz

from RAG_Agent.domain.doc_processing_rules.document_processing import DocumentProfileResolver
from RAG_Agent.domain.value_objects.canonical_document import CanonicalDocument
from RAG_Agent.domain.value_objects.canonical_merge import merge_canonical_shards
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


class NativePdfPipeline:
    """Pipeline PDF nativo: preprocesado → Docling (por shards) → CanonicalDocument.

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
            source_path = cleaned_path
            logger.info("Preprocessed PDF ready for Docling: %s", source_path.name)

            page_count = count_pdf_pages(source_path)
            logger.info(
                "Ingest profile=%s pages=%d shard_size=%d file=%s",
                hw.name,
                page_count,
                shard_size,
                source_path.name,
            )

            if page_count <= shard_size:
                docling_document = self._extractor.extract(source_path)
                return self._normalizer.normalize(
                    docling_document,
                    source_path=path,
                    profile=profile,
                    parser_name="native_pdf_docling",
                )

            shards = self._write_page_shards(source_path, tmp, shard_size, page_count)
            logger.info("Split into %d shards for Docling", len(shards))

            parts: list[tuple[int, CanonicalDocument]] = []
            for index, (page_offset, shard_path) in enumerate(shards):
                logger.info(
                    "Docling shard %d/%d (pages %d-%d)",
                    index + 1,
                    len(shards),
                    page_offset + 1,
                    min(page_offset + shard_size, page_count),
                )
                docling_document = self._extractor.extract(shard_path)
                canonical = self._normalizer.normalize(
                    docling_document,
                    source_path=path,
                    profile=profile,
                    parser_name="native_pdf_docling",
                )
                parts.append((page_offset, canonical))

            extra = dict(profile.identity.metadata)
            extra["ingest_profile"] = hw.name
            extra["pages_per_shard"] = str(shard_size)

            return merge_canonical_shards(
                parts,
                source_path=path,
                profile_id=profile.rules.profile_id,
                parser_name="native_pdf_docling",
                build_sections=lambda blocks: self._normalizer.build_sections(
                    blocks, rules=profile.rules
                ),
                resolve_title=lambda sections, blocks, hint: self._normalizer.resolve_title(
                    sections, blocks, hint, rules=profile.rules
                ),
                title_hint=profile.identity.title_hint,
                extra=extra,
            )

    @staticmethod
    def _write_page_shards(
        pdf_path: Path,
        output_dir: Path,
        shard_size: int,
        page_count: int,
    ) -> list[tuple[int, Path]]:
        """Escribe shards ``[offset, offset+shard_size)`` (índices 0-based)."""
        shards: list[tuple[int, Path]] = []
        with fitz.open(pdf_path) as src:
            for start in range(0, page_count, shard_size):
                end = min(start + shard_size, page_count) - 1
                shard_path = output_dir / f"{pdf_path.stem}_p{start + 1}-{end + 1}.pdf"
                shard = fitz.open()
                shard.insert_pdf(src, from_page=start, to_page=end)
                shard.save(shard_path)
                shard.close()
                shards.append((start, shard_path))
        return shards
