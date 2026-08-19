from typing import Dict, Any
from abc import ABC, abstractmethod
from pathlib import Path

from ..errors import ConfigError
from ..enums import FileType, Backend, ChunkingMethod
from .config import (
    BaseConfig,
    FileTypeConfig,
    BackendConfig,
    PathsConfig,
    IndexingConfig,
    RetrievalConfig,
    SearchConfig,
    AnswerConfig,
    Config
)


class BaseConfigFactory(ABC):
    @staticmethod
    @abstractmethod
    def create(raw_config: Dict[str, Any]) -> BaseConfig:
        pass


class FileTypeConfigFactory(BaseConfigFactory):
    @staticmethod
    def create(raw_config: Dict[str, Any]) -> FileTypeConfig:
        try:
            file_types = FileTypeConfig(
                chunking_method=ChunkingMethod(raw_config["chunking_method"]),
                enabled=raw_config["enabled"]
            )

        except Exception as e:
            raise ConfigError(
                f"An error has occured with file_types config:\n{e}"
            )

        return file_types


class BackendConfigFactory(BaseConfigFactory):
    @staticmethod
    def create(raw_config: Dict[str, Any]) -> BackendConfig:
        try:
            backend = BackendConfig(
                enabled=raw_config["enabled"],
                weight=raw_config["weight"]
            )

        except Exception as e:
            raise ConfigError(
                f"An error has occured with backend config:\n{e}"
            )

        return backend


class PathsConfigFactory(BaseConfigFactory):
    @staticmethod
    def create(
        raw_config: Dict[str, Any],
    ) -> PathsConfig:
        try:
            paths = PathsConfig(
                raw=Path(raw_config["raw"]),
                processed=Path(raw_config["processed"]),
                datasets=Path(raw_config["datasets"]),
                output=Path(raw_config["output"])
            )

        except Exception as e:
            raise ConfigError(
                f"An error has occured with paths config:\n{e}"
            )

        return paths


class IndexingConfigFactory(BaseConfigFactory):
    @staticmethod
    def create(raw_config: Dict[str, Any]) -> IndexingConfig:
        try:
            indexing = IndexingConfig(
                excludes=raw_config["excludes"],
                file_types={
                    FileType(key): FileTypeConfigFactory.create(
                        raw_config["file_types"][key]
                    ) for key in raw_config["file_types"]
                }
            )

        except Exception as e:
            raise ConfigError(
                f"An error has occured with indexing config:\n{e}"
            )

        return indexing


class RetrievalConfigFactory(BaseConfigFactory):
    @staticmethod
    def create(raw_config: Dict[str, Any]):
        try:
            retrieval = RetrievalConfig(
                backends={
                    Backend(key): BackendConfigFactory.create(
                        raw_config=raw_config["backends"][key]
                    ) for key in raw_config["backends"]
                }
            )

        except Exception as e:
            raise ConfigError(
                f"An error has occured with retrieval config:\n{e}"
            )

        return retrieval


class SearchConfigFactory(BaseConfigFactory):
    @staticmethod
    def create(raw_config: Dict[str, Any]):
        try:
            search = SearchConfig(
                auto_translate=raw_config["auto_translate"]
            )

        except Exception as e:
            raise ConfigError(
                f"An error has occured with search config:\n{e}"
            )

        return search


class AnswerConfigFactory(BaseConfigFactory):
    @staticmethod
    def create(raw_config: Dict[str, Any]):
        try:
            answer = AnswerConfig(
                model_name=raw_config["model_name"],
                max_context_length=raw_config["max_context_length"]
            )

        except Exception as e:
            raise ConfigError(
                f"An error has occured with answer config:\n{e}"
            )

        return answer


class ConfigFactory(BaseConfigFactory):
    @staticmethod
    def create(raw_config: Dict[str, Any]):
        try:
            config = Config(
                paths=PathsConfigFactory.create(
                    raw_config=raw_config["paths"]
                ),
                indexing=IndexingConfigFactory.create(
                    raw_config=raw_config["indexing"]
                ),
                retrieval=RetrievalConfigFactory.create(
                    raw_config=raw_config["retrieval"]
                ),
                search=SearchConfigFactory.create(
                    raw_config=raw_config["search"]
                ),
                answer=AnswerConfigFactory.create(
                    raw_config=raw_config["answer"]
                )
            )

        except Exception as e:
            raise ConfigError(
                f"An error has occured with the config:\n{e}"
            )

        return config
