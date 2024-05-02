import re
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException

class BooleanOutputParser(BaseOutputParser[bool]):
    """Custom boolean parser."""

    true_val: str = "YES"
    false_val: str = "NO"

    def parse(self, text: str) -> bool:
        found_bool_vals = re.findall(r"\bYES\b|\bNO\b", text.upper())

        if len(found_bool_vals) != 1:
            raise OutputParserException(
                f"BooleanOutputParser expected output value to either be "
                f"{self.true_val} or {self.false_val} (case-insensitive). "
                f"Received {text}."
            )
        return found_bool_vals[0] == self.true_val

    @property
    def _type(self) -> str:
        return "boolean_output_parser"