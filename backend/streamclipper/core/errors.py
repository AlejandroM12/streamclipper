class StreamClipperError(Exception):
    """Error esperado, con mensaje apto para mostrar al usuario."""


class InvalidInputError(StreamClipperError):
    pass


class MediaError(StreamClipperError):
    pass


class ToolNotFoundError(StreamClipperError):
    pass


class CancelledError(StreamClipperError):
    pass
