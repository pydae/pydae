"""
pydae.utils — miscellaneous helpers.

Re-exports the public API of the individual submodules so users can write::

    from pydae.utils import read_data, save_json, save_hjson
    from pydae.utils import convert_all_svgs_to_pdf
    from pydae.utils import get_absolute_hour_of_year_with_dst

instead of reaching into the submodule paths.
"""

from pydae.utils.utils import (
    read_data,
    save_json,
    save_hjson,
)

from pydae.utils.svg2pdf import (
    convert_all_svgs_to_pdf,
    load_converted_data,
    save_converted_data,
    get_file_mod_time,
)

from pydae.utils.dates import (
    get_absolute_hour_of_year_with_dst,
)

__all__ = [
    # utils.py
    "read_data",
    "save_json",
    "save_hjson",
    # svg2pdf.py
    "convert_all_svgs_to_pdf",
    "load_converted_data",
    "save_converted_data",
    "get_file_mod_time",
    # dates.py
    "get_absolute_hour_of_year_with_dst",
]
