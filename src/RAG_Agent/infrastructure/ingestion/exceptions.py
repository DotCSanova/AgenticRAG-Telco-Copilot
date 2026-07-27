class PDFParsingException(Exception):
    """Error base del pipeline de parsing PDF."""


class PDFValidationError(PDFParsingException):
    """El PDF no cumple validaciones previas al parsing."""
