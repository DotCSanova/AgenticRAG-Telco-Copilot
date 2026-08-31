from types import SimpleNamespace

from RAG_Agent.infrastructure.ingestion.normalizers.docling_normalizer import (
    DoclingNormalizer,
    _resolve_caption_text,
)


class _Cell:
    def __init__(self, text: str, *, column_header: bool = False) -> None:
        self.text = text
        self.column_header = column_header


def test_grid_uses_column_header_rows():
    table = SimpleNamespace(
        self_ref="#/tables/0",
        data=SimpleNamespace(
            grid=[
                [_Cell("Level", column_header=True), _Cell("Priority", column_header=True)],
                [_Cell("1"), _Cell("Very HIGH")],
            ]
        ),
    )
    data = DoclingNormalizer._table_to_data(table)
    assert data.headers == ["Level", "Priority"]
    assert data.rows == [["1", "Very HIGH"]]


def test_grid_without_column_header_keeps_all_rows():
    table = SimpleNamespace(
        self_ref="#/tables/1",
        data=SimpleNamespace(
            grid=[
                [_Cell("0"), _Cell("1")],
                [_Cell("a"), _Cell("b")],
            ]
        ),
    )
    data = DoclingNormalizer._table_to_data(table)
    assert data.headers == []
    assert data.rows == [["0", "1"], ["a", "b"]]


def test_broken_grid_degrades_to_empty_table():
    class _Boom:
        @property
        def grid(self) -> list:
            raise ValueError("no cells")

    data = DoclingNormalizer._table_to_data(SimpleNamespace(self_ref="#/tables/9", data=_Boom()))
    assert data.headers == []
    assert data.rows == []


def test_resolve_caption_uses_ref_text_not_cref_string():
    class _Ref:
        cref = "#/texts/255"

        def resolve(self, doc: object) -> SimpleNamespace:
            return SimpleNamespace(text="Figure A.1-1: Reduction of environmental impact")

    text = _resolve_caption_text(object(), [_Ref()])
    assert text == "Figure A.1-1: Reduction of environmental impact"
    assert "cref" not in text
