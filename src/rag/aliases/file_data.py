from typing import TypeAlias

from ..enums import FileType


FileData: TypeAlias = tuple[FileType, str, str]
