"""Lexicons as user-editable files.

Every file in the lexicon directory (``config.yaml`` -> ``lexicons_dir``)
becomes one lexicon, named after the file:

* ``german_cities.csv``  -> lexicon ``german_cities`` (first CSV column,
  or the column named ``name`` if present)
* ``my_departments.txt`` -> lexicon ``my_departments`` (one entry per line)

To change a lexicon, edit the file; to add one, drop a new file in and pick
its name in the field table's ``lexicon`` column. No code changes.

Loading itself is htrpipe's (``PostprocessResources.load_lexicon_from_csv`` /
``load_lexicon_from_lines``); this module only walks the directory.

The two rules that use word lists without calling them lexicons live here
too: ``read_list`` feeds ``email_domains.txt`` (the ``email`` rule) and
``street_suffixes.txt`` (the ``street_suffix`` rule) from the resources
folder, so all three lists are user-editable files.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List


def read_list(path) -> List[str]:
    """Order-preserving list file: one entry per line, ``#`` comments and
    blank lines skipped, duplicates dropped keeping the first occurrence.

    Order is preserved deliberately -- the street-suffix corrector breaks
    distance ties in favour of earlier entries, so sorting the file would
    subtly change which correction wins.
    """
    seen = set()
    out: List[str] = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry in seen:
            continue
        seen.add(entry)
        out.append(entry)
    return out


def available_lexicons(lexicons_dir) -> List[str]:
    """Lexicon names on offer -- the file stems in the directory."""
    d = pathlib.Path(lexicons_dir)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.iterdir()
                  if p.suffix.lower() in (".csv", ".txt") and p.is_file())


def load_into(resources, lexicons_dir) -> Dict[str, int]:
    """Load every lexicon file into ``resources.lexicons``.

    Returns {lexicon name: entry count} for display and the manifest.
    CSVs load their ``name`` column when it exists, otherwise the first
    column; ``.txt`` files load one entry per line.
    """
    import pandas as pd

    counts: Dict[str, int] = {}
    d = pathlib.Path(lexicons_dir)
    if not d.is_dir():
        return counts
    for path in sorted(d.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".csv":
            columns = pd.read_csv(path, nrows=0).columns
            column = "name" if "name" in columns else columns[0]
            counts[path.stem] = resources.load_lexicon_from_csv(
                path.stem, str(path), column)
        elif path.suffix.lower() == ".txt":
            counts[path.stem] = resources.load_lexicon_from_lines(
                path.stem, str(path))
    return counts
