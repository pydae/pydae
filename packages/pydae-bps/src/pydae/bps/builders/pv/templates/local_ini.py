#!/home/jmmmacubu/miniforge3/envs/pydae_dev/bin/python3
"""Build the pydae model from the plant JSON and call ``Model.ini({})``.

Minimal harness to reproduce the initialisation step in isolation, without
Mininet, without proxies, without FastAPI.  Useful for debugging IDAS /
Newton convergence when the cosim fails at boot::

    ./ini.py                 # uses the plant JSON next to this script
    ./ini.py -j other.json

The initial guess ``xy_0.json`` is expected to be produced during the bps
building stage (``grid.construct``); this script does NOT synthesise one.
If ``ini()`` fails the traceback is printed and the exit code is non-zero.
"""
from __future__ import annotations

import argparse
import os
import sys

from pydae.bps import BpsBuilder
from pydae.core.builder import CasadiBuilder, CasadiModel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", "-j",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "__JSON_FILENAME__"),
                    help="Plant JSON path")
    args = ap.parse_args()

    json_path = os.path.abspath(args.json)
    name = os.path.splitext(os.path.basename(json_path))[0]
    if not os.path.exists(json_path):
        print(f"error: JSON not found: {json_path}", file=sys.stderr)
        return 2

    # construct() writes build artefacts (xy_0.json etc.) to the cwd;
    # keep them next to the JSON, mirroring pydae_sim's convention.
    os.chdir(os.path.dirname(json_path))

    print(f"[ini] building bps grid from {json_path}")
    grid = BpsBuilder(json_path, use_casadi=True)
    grid.checker()
    grid.uz_jacs = False
    grid.construct(name)

    print(f"[ini] wrapping CasadiBuilder -> CasadiModel")
    model = CasadiModel(CasadiBuilder(grid.sys_dict).build())
    model.report_u()

    print(f"[ini] calling model.ini({{}}) …")
    model.ini({})
    print(f"[ini] OK — initialisation converged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
