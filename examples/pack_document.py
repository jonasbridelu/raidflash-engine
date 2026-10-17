#!/usr/bin/env python3
"""v0.1: pack one document into --parallel sister streams after a drafted seed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from raidflash import Packer

DOC = (
    "Acta de reunion del comite tecnico. Disponibilidad 99.97 por ciento. "
    "Dos incidentes menores. Auditoria sin hallazgos criticos."
)
SECTIONS = [
    "Estado del cluster",
    "Almacenamiento",
    "Contingencia electrica",
    "Auditoria de seguridad",
]


def main() -> None:
    base = os.environ.get("RAIDFLASH_BASE_URL", "http://192.168.100.11:8000")
    p = Packer(base_url=base)
    out = p.complete(DOC, SECTIONS, stripe_tokens=80)
    print(f"k={out['k']} wall={out['wall_s']}s agg={out['agg_tok_s']} "
          f"cached_min={out['cached_pct_min']}%")
    print(out["text"][:800])


if __name__ == "__main__":
    main()
