#!/usr/bin/env python3
"""
Scrape electron-impact excitation cross sections from the Curtin CCC database.

Target use:
- Hydrogen (H I) by default, but configurable via CLI arguments.
- Outputs a tidy CSV with columns:
  LowerState, UpperState, Transition, Energy_eV, CrossSection_cm2

Notes:
- This script performs polite scraping with rate limiting.
- The CCC site may change HTML structure or request parameters over time.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import requests
from bs4 import BeautifulSoup


LOG = logging.getLogger(__name__)

CCC_URL = "https://atom.curtin.edu.au/cgi-bin/c4nick.in"

NUM_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")

# Maps orbital letter -> CCC term letter
TERM_MAP = {"s": "S", "p": "P", "d": "D", "f": "F"}
L_MAP = {"s": 0, "p": 1, "d": 2, "f": 3}


@dataclass(frozen=True)
class State:
    """Atomic state represented by principal quantum number n and orbital letter l (s,p,d,f)."""
    n: int
    l: str  # 's','p','d','f'

    def __post_init__(self) -> None:
        l_norm = self.l.lower().strip()
        if l_norm not in TERM_MAP:
            raise ValueError(f"Invalid l='{self.l}'. Must be one of {sorted(TERM_MAP)}.")
        object.__setattr__(self, "l", l_norm)
        if self.n <= 0:
            raise ValueError("n must be a positive integer.")

    @property
    def label(self) -> str:
        return f"{self.n}{self.l}"

    @property
    def energy_index(self) -> int:
        """
        Sorting key used to avoid de-excitation requests.
        Not a physical energy model—just a consistent ordering: n*10 + l_value.
        """
        return self.n * 10 + L_MAP[self.l]

    def ccc_code(self) -> str:
        """
        Generate CCC database state code.
        Example: n=2,l=p -> '2p_dPz2pzX2YP'
        """
        term = TERM_MAP[self.l]
        return f"{self.n}{self.l}_d{term}z{self.n}{self.l}zX2Y{term}"


def parse_states_list(spec: str) -> List[State]:
    """
    Parse comma-separated state spec like:
      "1s,2s,2p,3s,3p,3d,4s,4p,4d,4f"
    """
    out: List[State] = []
    for token in [t.strip() for t in spec.split(",") if t.strip()]:
        m = re.fullmatch(r"(\d+)\s*([spdfSPDF])", token)
        if not m:
            raise ValueError(f"Bad state token '{token}'. Expected like '2p'.")
        n = int(m.group(1))
        l = m.group(2).lower()
        out.append(State(n=n, l=l))
    return out


def fetch_transition_data(
    session: requests.Session,
    element: str,
    charge: str,
    low: State,
    upp: State,
    timeout_s: float,
) -> List[Tuple[float, float]]:
    """
    Fetch (Energy_eV, CrossSection_cm2) points for one transition from CCC.

    Returns:
        List of (energy_eV, cross_section_cm2) points.
        Empty list if no data table is found or parsing yields nothing.
    """
    params = {
        "element": element,
        "charge": charge,
        "process": "ELECTRON_IMPACT",
        "low": low.ccc_code(),
        "upp": upp.ccc_code(),
        "data_type": "cross",
        "cs_unit": "cm2",
    }

    headers = {"User-Agent": "Mozilla/5.0 (compatible; CCC-scraper/1.0)"}

    r = session.get(CCC_URL, params=params, headers=headers, timeout=timeout_s)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table")
    if table is None:
        return []

    points: List[Tuple[float, float]] = []
    for tr in table.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) < 2:
            continue

        e_match = NUM_RE.search(tds[0])
        s_match = NUM_RE.search(tds[-1])
        if not (e_match and s_match):
            continue

        try:
            e = float(e_match.group())
            sigma = float(s_match.group())
        except ValueError:
            continue

        points.append((e, sigma))

    return points


def iter_allowed_transitions(
    lower_states: Sequence[State],
    upper_states: Sequence[State],
    allow_deexcitation: bool,
) -> Iterable[Tuple[State, State]]:
    """
    Generate transitions while skipping:
    - self-transitions always
    - de-excitation transitions unless allow_deexcitation is True
    """
    for low in lower_states:
        for upp in upper_states:
            if low == upp:
                continue
            if (not allow_deexcitation) and (low.energy_index > upp.energy_index):
                continue
            yield low, upp


def write_csv(
    outpath: Path,
    rows: Iterable[Tuple[str, str, str, float, float]],
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["LowerState", "UpperState", "Transition", "Energy_eV", "CrossSection_cm2"])
        for row in rows:
            w.writerow(row)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scrape Curtin CCC electron-impact excitation cross sections and write CSV."
    )
    p.add_argument("--element", default="H", help="Element symbol, e.g. H, He, etc. (default: H)")
    p.add_argument("--charge", default="I", help="Charge state as CCC expects (default: I)")
    p.add_argument(
        "--lower",
        default="1s,2s,2p",
        help="Comma-separated lower states (default: 1s,2s,2p)",
    )
    p.add_argument(
        "--upper",
        default="1s,2s,2p,3s,3p,3d,4s,4p,4d,4f",
        help="Comma-separated upper states (default: 1s..4f as in your list)",
    )
    p.add_argument(
        "--outdir",
        default=".",
        help="Output directory (default: current directory).",
    )
    p.add_argument(
        "--outfile",
        default="ccc_cross_sections.csv",
        help="Output CSV filename (default: ccc_cross_sections.csv)",
    )
    p.add_argument(
        "--delay-ok",
        type=float,
        default=0.5,
        help="Delay (seconds) after successful fetch (default: 0.5)",
    )
    p.add_argument(
        "--delay-nodata",
        type=float,
        default=0.2,
        help="Delay (seconds) after 'no data' fetch (default: 0.2)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds (default: 10)",
    )
    p.add_argument(
        "--allow-deexcitation",
        action="store_true",
        help="If set, do not skip low->upp ordering by energy_index.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    lower_states = parse_states_list(args.lower)
    upper_states = parse_states_list(args.upper)
# CSV file location
    script_dir = Path(__file__).resolve().parent
    outdir = script_dir
    outpath = outdir / args.outfile

    LOG.info("CCC scrape starting: %s %s", args.element, args.charge)
    LOG.info("Lower states: %s", ", ".join(s.label for s in lower_states))
    LOG.info("Upper states: %s", ", ".join(s.label for s in upper_states))
    LOG.info("Output: %s", outpath)

    total_points = 0
    csv_rows: List[Tuple[str, str, str, float, float]] = []

    with requests.Session() as session:
        for low, upp in iter_allowed_transitions(lower_states, upper_states, args.allow_deexcitation):
            trans_label = f"{low.label} -> {upp.label}"

            LOG.info("Processing %s", trans_label)
            try:
                points = fetch_transition_data(
                    session=session,
                    element=args.element,
                    charge=args.charge,
                    low=low,
                    upp=upp,
                    timeout_s=args.timeout,
                )
            except requests.RequestException as e:
                LOG.warning("Request failed for %s: %s", trans_label, e)
                # still be polite even on errors
                time.sleep(args.delay_nodata)
                continue

            if not points:
                LOG.info("No data for %s", trans_label)
                time.sleep(args.delay_nodata)
                continue

            LOG.info("OK: %d points for %s", len(points), trans_label)
            for energy, sigma in points:
                csv_rows.append((low.label, upp.label, trans_label, energy, sigma))
            total_points += len(points)

            time.sleep(args.delay_ok)

    write_csv(outpath, csv_rows)
    LOG.info("Done. Total points collected: %d", total_points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
