import json
import logging
import sys
import time
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        data.update(getattr(record, "fields", {}))
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def get_logger() -> logging.Logger:
    return logging.getLogger("streamclipper")


def setup_logging(log_file: Path | None = None, verbose: bool = False) -> None:
    logger = get_logger()
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(JsonFormatter())
    logger.addHandler(console)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        logger.addHandler(fh)


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    get_logger().log(level, event, extra={"fields": fields})
