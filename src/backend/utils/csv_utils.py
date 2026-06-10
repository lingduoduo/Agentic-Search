import csv
import io
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class ParsedRow:
    header: list[str]
    row: list[str]


def read_csv_header(csv_string: str) -> list[str]:
    reader = csv.reader(io.StringIO(csv_string))
    try:
        return next(reader)
    except StopIteration:
        return []


def parse_csv_string(csv_string: str) -> Iterator[ParsedRow]:
    reader = csv.reader(io.StringIO(csv_string))
    try:
        header = next(reader)
    except StopIteration:
        return
    for row in reader:
        yield ParsedRow(header=header, row=row)
