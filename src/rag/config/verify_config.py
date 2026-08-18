from typing import Dict, Any

from ..errors import ConfigError

SCHEMA = {
    "paths": {
        "raw": "foo",
        "processed": "bar",
        "datasets": "baz",
        "output": "qux"
    },

    "indexing": {
        "excludes": ["foo"],
        "file_types": {
            ".py": {"chunking_method": "foo", "enabled": False},
            ".md": {"chunking_method": "foo", "enabled": False},
            ".txt": {"chunking_method": "foo", "enabled": False},
        }
    },

    "retrieval": {
        "backends": {
            "tfidf": {"enabled": False},
            "bm25": {"enabled": False},
            "chroma": {"enabled": False}
        }
    },

    "search": {"auto_translate": False},

    "answer": {"model_name": "foo", "max_context_length": 67}
}


def recursive_check(
    config_branch: Any,
    schema_branch: Any,
) -> None:
    """

    """
    if not isinstance(config_branch, type(schema_branch)):
        raise ConfigError(
            f"wrong data type: {type(config_branch)}"
            + f", should be {type(schema_branch)}"
        )

    if not isinstance(config_branch, (dict, list)):
        return

    if isinstance(config_branch, list):
        for element in config_branch:
            recursive_check(element, schema_branch[0])
        return

    assert isinstance(schema_branch, dict)

    for config_key in config_branch.keys():
        if config_key not in schema_branch:
            raise ConfigError(f"Invalid key: {config_key}")
        recursive_check(
            config_branch[config_key],
            schema_branch[config_key],
        )

    if len(config_branch) < len(schema_branch):
        raise ConfigError(
            "missing key(s): " + str(list(
                key for key in schema_branch.keys()
                if key not in config_branch
            ))
        )


def verify_config_schema(raw_config: Dict) -> None:
    recursive_check(raw_config, SCHEMA)
