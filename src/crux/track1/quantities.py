"""Extract numeric quantities (value + optional unit) from text and compare them, for the L1 numeric check (P4).

Handles thousands separators (50,300), decimals, fractions (3/7), scientific notation (5.03e4, 5.03 x 10^4,
5.03 × 10⁴) and SI prefixes on a small set of base units (kg/mole == 1000 g/mol). Numbers glued to letters,
hyphens or comma-separated digit lists are skipped, so chemistry locants ("2,3-diphenyl", "(1S,2S,5R)") and
names like "R-loops" do not produce quantities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUPER_EXP_RE = re.compile(r"10\s*([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)")

_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+"
_UNIT = r"[A-Za-zµμ°%Å]+(?:\s*/\s*[A-Za-zµμ]+)?"
_QUANTITY_RE = re.compile(
    rf"(?<![\w.,/^\-])(?P<sign>-)?(?P<num>{_NUMBER})"
    rf"(?:\s*/\s*(?P<den>\d+)(?![\d.]))?"
    rf"(?:\s*[x×*]\s*10\s*\^\s*(?P<exp>[-+]?\d+)|[eE](?P<exp2>[-+]?\d+))?"
    rf"(?:\s?(?P<unit>{_UNIT}))?"
    rf"(?![\w\-]|,\d|\.\d)"
)

BASE_UNITS = {"g", "m", "s", "L", "mol", "J", "eV", "Hz", "Pa", "W", "V", "N", "M", "Da", "A", "Wb", "T"}
PREFIXES = {"k": 1e3, "M": 1e6, "G": 1e9, "c": 1e-2, "m": 1e-3, "µ": 1e-6, "μ": 1e-6, "u": 1e-6, "n": 1e-9,
            "p": 1e-12}
UNIT_ALIASES = {"mole": "mol", "moles": "mol", "gram": "g", "grams": "g", "sec": "s", "second": "s",
                "seconds": "s", "liter": "L", "litre": "L", "liters": "L", "l": "L", "ml": "mL", "kda": "kDa"}


@dataclass(frozen=True)
class Quantity:
    value: float              # in base units when the unit is known, else as written
    unit: Optional[str]       # canonical base unit ("g/mol"), the raw unit if unknown, or None
    unit_known: bool
    raw: str


def _canonical_part(unit: str) -> Optional[Tuple[float, str]]:
    unit = UNIT_ALIASES.get(unit, UNIT_ALIASES.get(unit.lower(), unit))
    if unit in BASE_UNITS:
        return 1.0, unit
    if len(unit) > 1 and unit[0] in PREFIXES and unit[1:] in BASE_UNITS:
        return PREFIXES[unit[0]], unit[1:]
    return None


def canonical_unit(unit: str) -> Optional[Tuple[float, str]]:
    """(factor to base, base unit) for e.g. "kg/mole" -> (1000.0, "g/mol"); None if any part is unknown."""
    parts = [part.strip() for part in unit.split("/")]
    canon = [_canonical_part(part) for part in parts]
    if any(c is None for c in canon):
        return None
    factor = canon[0][0] / (canon[1][0] if len(canon) > 1 else 1.0)
    return factor, "/".join(c[1] for c in canon)


def normalize_text(text: str) -> str:
    text = _SUPER_EXP_RE.sub(lambda m: "10^" + m.group(1).translate(_SUPERSCRIPTS), text)
    return text.replace("−", "-").replace(" ", " ").replace(" ", " ")


def extract_quantities(text: str) -> List[Quantity]:
    quantities = []
    for match in _QUANTITY_RE.finditer(normalize_text(text)):
        value = float(match.group("num").replace(",", ""))
        if match.group("den"):
            denominator = float(match.group("den"))
            if denominator == 0:
                continue
            value /= denominator
        exponent = match.group("exp") or match.group("exp2")
        if exponent:
            value *= 10 ** int(exponent)
        if match.group("sign"):
            value = -value
        unit_raw = match.group("unit")
        unit, known = None, False
        if unit_raw:
            unit_raw = re.sub(r"\s+", "", unit_raw)
            canon = canonical_unit(unit_raw)
            if canon:
                value, unit, known = value * canon[0], canon[1], True
            else:
                unit = unit_raw
        quantities.append(Quantity(value=value, unit=unit, unit_known=known, raw=match.group(0).strip()))
    return quantities


def relative_difference(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0 else abs(a - b) / scale


def comparable(a: Quantity, b: Quantity) -> bool:
    """Two known units must share a base; otherwise compare the numbers as written."""
    if a.unit_known and b.unit_known:
        return a.unit == b.unit
    return True
