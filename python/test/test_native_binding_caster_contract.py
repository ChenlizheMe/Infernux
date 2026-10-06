"""One caster definition per native type is a C++ ABI requirement across binding units."""
from pathlib import Path
import re


def test_every_binding_translation_unit_loads_the_common_caster_contract():
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / "cpp/infernux/tools/pybinding").glob("Binding*.cpp"))
    assert sources
    missing = [source.name for source in sources
               if not re.search(r'^#include "BindingRegistration\.h"$',
                                source.read_text(encoding="utf-8"), re.MULTILINE)]
    assert not missing, f"Native caster definitions would differ across translation units: {missing}"
