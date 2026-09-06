import logging
import shutil
from typing import Optional

try:
    import pysfacpp_internal  # ty:ignore[unresolved-import]
except ImportError:
    pysfacpp_internal = None  # type: ignore[assignment]


class PySfaCppBase:

    def __init__(self) -> None:
        pass

    def __setup_logger(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] - %(thread)d - [pysfacpp] %(levelname)s: %(message)s',
            datefmt="%H:%M:%S",
            force=True
        )
        # register the Logger
        try:
            pysfacpp_internal._register_logger(self.__spdlog_adapter)
        except AttributeError:
            # handles case where the C++ module might be outdated and lacks the function
            logging.getLogger(name=__name__).warning(
                msg="C++ logger registration failed: 'pysfacpp_internal' has no 'register_logger' function."
            )
        except Exception as e:
            logging.getLogger(name=__name__).warning(
                msg=f"Failed to attach C++ spdlog to Python logger: {e}"
            )
        # add null handler if no logging configured
        logging.getLogger(name=__name__).addHandler(hdlr=logging.NullHandler())
    
    def __spdlog_adapter(self, logger_name: str, msg: str, level: int) -> None:
        level_map: dict[int, int] = {
            0: logging.DEBUG,
            1: logging.DEBUG,
            2: logging.INFO,
            3: logging.WARNING,
            4: logging.ERROR,
            5: logging.CRITICAL,
        }
        py_level: int = level_map.get(level, logging.INFO)
        logger: logging.Logger = logging.getLogger(name=__name__)
        if logger_name == "raw":
            print(msg)
        else:
            logger.log(py_level, msg)

    def __console_width(self) -> int:
        cols, _ = shutil.get_terminal_size(fallback=(120, 80))
        return min(cols, 150)

    def __default_decimals(self) -> int:
        return 5
