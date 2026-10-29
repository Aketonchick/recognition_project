import os.path
from pathlib import Path
from os import PathLike
import json

class Counter:

    _all_counters: dict[str, int] = {}

    def __init__(self, filename: PathLike, name: str):
        self.filename = Path(filename)
        self.name = name
        self.count = 0
        self.current_value = self._get_current_value(self.name)

    def __call__(self, increment: int = 1):
        self.current_value += increment
        self._all_counters[self.name] += increment

        with open(self.filename, "w") as f:
            json.dump(self._all_counters, f)

    def _get_current_value(self, name) -> int:
        self._load_from_file()

        if name not in self._all_counters:
            self._all_counters[name] = 0
        current_value = self._all_counters[name]

        return current_value

    def _load_from_file(self) -> None:
        """Загружает данные из файла в кеш"""
        file_name = str(self.filename)

        if not os.path.exists(file_name):
            self._all_counters = {}
            return

        try:
            with open(file_name, "r") as f:
                self._all_counters = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Файл содержит неверную структуру или он пустой: {e}")

    def get_value(self) -> int:
        return self.current_value



