import codeop
from pathlib import Path
from typing import Any

import duckdb
from loguru import logger

from karldbot.models.config import ProblemConfig

logger.remove(0)
logger.add("code_bugs.log", rotation="1 MB", level="INFO")


class DataScienceProblem:
    def __init__(self, config: ProblemConfig):
        self.config = config
        self.problem_name = config.name
        self.data_source = str(config.data_source)
        self.description = config.description
        self.data_table = "data"
        self.data_loaded = False
        self._connection: duckdb.DuckDBPyConnection | None = None
        self.load_data()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataScienceProblem":
        config = ProblemConfig.from_yaml(path)
        return cls(config)

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        if self._connection is None:
            self._connection = duckdb.connect(":memory:")
        return self._connection

    def load_data(self) -> None:
        self.connection.execute(
            f"CREATE TABLE {self.data_table} AS "
            f"SELECT * FROM read_csv_auto('{self.data_source}')"
        )
        self.data_loaded = True

    def sample_data(self, n: int = 10) -> str:
        result = self.connection.execute(
            f"SELECT * FROM {self.data_table} LIMIT {n}"
        ).fetchdf()
        return result.to_markdown()

    def get_schema(self) -> str:
        result = self.connection.execute(f"DESCRIBE {self.data_table}").fetchdf()
        return result.to_markdown()

    def evaluate_solution(self, solution: str) -> dict[str, Any]:
        raise NotImplementedError("Subclasses must implement evaluate_solution")

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
