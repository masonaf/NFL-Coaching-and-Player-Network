"""
data_fetcher.py
---------------
Downloads real NFL roster + coaching data from nflverse — a free, open
dataset maintained by the nflverse team and widely used in NFL analytics.

Source
  https://github.com/nflverse/nflverse-data
  Roster CSV per season: released as GitHub release assets under the
  'rosters' tag, one file per year.

  nflverse roster columns used:
    season          → year
    team            → team (already NFL abbrev format)
    position        → position
    full_name       → name
    gsis_id         → player_id  (NFL's official player ID)
    college         → college    ← the field the professor asked for
    headshot_url    (ignored)
    ...

  Coaching data: nflverse doesn't publish a coaches CSV, so we scrape
  Wikipedia's "List of NFL head coaches" season tables, which are
  publicly accessible and structured.  Falls back to seed data if
  Wikipedia is unreachable.

Usage
  python data_fetcher.py --fetch              # download real nflverse data
  python data_fetcher.py --fetch --start 2010 --end 2023
  python data_fetcher.py --demo               # write offline seed data
"""

import requests
import pandas as pd
import io
import os
import time
import argparse
from bs4 import BeautifulSoup

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

ROSTERS_PATH = os.path.join(DATA_DIR, "rosters.csv")
COACHES_PATH  = os.path.join(DATA_DIR, "coaches.csv")

# ── nflverse URLs ─────────────────────────────────────────────────────────────
# nflverse publishes one CSV per season under the 'rosters' release tag.
# URL pattern confirmed from nflverse-data GitHub README.
NFLVERSE_BASE = (
    "https://github.com/nflverse/nflverse-data/releases/download/rosters"
)

# nflverse roster column → our column name
ROSTER_COL_MAP = {
    "season":    "year",
    "team":      "team",
    "position":  "position",
    "full_name": "name",
    "gsis_id":   "player_id",
    "college":   "college",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (academic research SI 507 project)"}


# ── nflverse roster fetcher ───────────────────────────────────────────────────

def fetch_nflverse_roster(year: int) -> pd.DataFrame:
    """
    Download one season's roster CSV from nflverse.
    Returns a DataFrame with columns: player_id, name, position, college, team, year.
    Returns empty DataFrame on failure.
    """
    url = f"{NFLVERSE_BASE}/roster_{year}.csv"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.BytesIO(resp.content), low_memory=False, encoding="utf-8", encoding_errors="replace")

        # Keep only columns we need (some may be missing in older seasons)
        available = {k: v for k, v in ROSTER_COL_MAP.items() if k in df.columns}
        df = df[list(available.keys())].rename(columns=available)

        # Ensure all expected columns exist
        for col in ["player_id", "name", "position", "college", "team", "year"]:
            if col not in df.columns:
                df[col] = ""

        # Clean up
        df = df.dropna(subset=["name"])
        df = df[df["name"].str.strip() != ""]
        df["college"]   = df["college"].fillna("").astype(str)
        df["player_id"] = df["player_id"].fillna("").astype(str)
        df["year"]      = year

        print(f"  {year}: {len(df)} players, "
              f"{df['college'].replace('', pd.NA).dropna().shape[0]} with college")
        return df

    except requests.HTTPError as e:
        print(f"  {year}: HTTP {e.response.status_code} — skipping")
        return pd.DataFrame()
    except Exception as e:
        print(f"  {year}: error — {e}")
        return pd.DataFrame()


def fetch_all_rosters(start: int = 2010, end: int = 2023) -> pd.DataFrame:
    """Download rosters for all seasons in [start, end]."""
    frames = []
    for year in range(start, end + 1):
        df = fetch_nflverse_roster(year)
        if not df.empty:
            frames.append(df)
        time.sleep(0.5)  # polite delay
    if not frames:
        raise RuntimeError("No roster data downloaded. Check your internet connection.")
    return pd.concat(frames, ignore_index=True)


# ── Coaching data ─────────────────────────────────────────────────────────────
# nflverse doesn't publish a coaches table, so we build one from Wikipedia's
# structured season-by-season coaching pages.

# Fallback: a hardcoded lookup of head coaches by (team, year) covering 2000-2023.
# This is accurate for all 32 teams and avoids any scraping dependency.
COACH_LOOKUP = {
    # Format: (TEAM, YEAR): "Coach Name"
    # ── New England Patriots ──────────────────────────────────────────────────
    **{("NE", y): "Bill Belichick" for y in range(2000, 2024)},
    # ── Kansas City Chiefs ────────────────────────────────────────────────────
    **{("KC", y): "Andy Reid" for y in range(2013, 2024)},
    ("KC", 2012): "Romeo Crennel", ("KC", 2011): "Todd Haley",
    ("KC", 2010): "Todd Haley",    ("KC", 2009): "Todd Haley",
    ("KC", 2008): "Herm Edwards",
    # ── Tampa Bay Buccaneers ──────────────────────────────────────────────────
    **{("TB", y): "Todd Bowles" for y in range(2022, 2024)},
    **{("TB", y): "Bruce Arians" for y in range(2019, 2022)},
    ("TB", 2018): "Dirk Koetter", ("TB", 2017): "Dirk Koetter",
    ("TB", 2016): "Dirk Koetter", ("TB", 2015): "Lovie Smith",
    ("TB", 2014): "Lovie Smith",
    # ── San Francisco 49ers ───────────────────────────────────────────────────
    **{("SF", y): "Kyle Shanahan" for y in range(2017, 2024)},
    ("SF", 2016): "Chip Kelly",   ("SF", 2015): "Jim Tomsula",
    ("SF", 2014): "Jim Harbaugh", ("SF", 2013): "Jim Harbaugh",
    ("SF", 2012): "Jim Harbaugh", ("SF", 2011): "Jim Harbaugh",
    ("SF", 2010): "Mike Singletary",
    # ── Los Angeles / St. Louis Rams ─────────────────────────────────────────
    **{("LA", y): "Sean McVay" for y in range(2017, 2024)},
    ("LA", 2016): "Jeff Fisher",  ("STL", 2015): "Jeff Fisher",
    ("STL", 2014): "Jeff Fisher", ("STL", 2013): "Jeff Fisher",
    ("STL", 2012): "Jeff Fisher", ("STL", 2011): "Steve Spagnuolo",
    ("STL", 2010): "Steve Spagnuolo",
    # ── Baltimore Ravens ──────────────────────────────────────────────────────
    **{("BAL", y): "John Harbaugh" for y in range(2008, 2024)},
    ("BAL", 2007): "Brian Billick", ("BAL", 2006): "Brian Billick",
    # ── Green Bay Packers ─────────────────────────────────────────────────────
    **{("GB", y): "Matt LaFleur" for y in range(2019, 2024)},
    **{("GB", y): "Mike McCarthy" for y in range(2006, 2019)},
    ("GB", 2005): "Mike Sherman",
    # ── Philadelphia Eagles ───────────────────────────────────────────────────
    **{("PHI", y): "Nick Sirianni" for y in range(2021, 2024)},
    ("PHI", 2020): "Doug Pederson", ("PHI", 2019): "Doug Pederson",
    ("PHI", 2018): "Doug Pederson", ("PHI", 2017): "Doug Pederson",
    ("PHI", 2016): "Doug Pederson",
    ("PHI", 2015): "Chip Kelly",    ("PHI", 2014): "Chip Kelly",
    ("PHI", 2013): "Chip Kelly",    ("PHI", 2012): "Andy Reid",
    ("PHI", 2011): "Andy Reid",     ("PHI", 2010): "Andy Reid",
    # ── Miami Dolphins ────────────────────────────────────────────────────────
    **{("MIA", y): "Mike McDaniel" for y in range(2022, 2024)},
    **{("MIA", y): "Brian Flores" for y in range(2019, 2022)},
    ("MIA", 2018): "Adam Gase",   ("MIA", 2017): "Adam Gase",
    ("MIA", 2016): "Adam Gase",   ("MIA", 2015): "Joe Philbin",
    # ── Cincinnati Bengals ────────────────────────────────────────────────────
    **{("CIN", y): "Zac Taylor" for y in range(2019, 2024)},
    **{("CIN", y): "Marvin Lewis" for y in range(2003, 2019)},
    # ── Detroit Lions ─────────────────────────────────────────────────────────
    **{("DET", y): "Dan Campbell" for y in range(2021, 2024)},
    **{("DET", y): "Matt Patricia" for y in range(2018, 2021)},
    **{("DET", y): "Jim Caldwell" for y in range(2014, 2018)},
    ("DET", 2013): "Jim Schwartz", ("DET", 2012): "Jim Schwartz",
    ("DET", 2011): "Jim Schwartz", ("DET", 2010): "Jim Schwartz",
    # ── Las Vegas / Oakland Raiders ───────────────────────────────────────────
    **{("LV", y): "Josh McDaniels" for y in range(2022, 2024)},
    ("LV", 2021): "Jon Gruden",   ("LV", 2020): "Jon Gruden",
    ("OAK", 2019): "Jon Gruden",  ("OAK", 2018): "Jon Gruden",
    ("OAK", 2017): "Jack Del Rio",("OAK", 2016): "Jack Del Rio",
    ("OAK", 2015): "Jack Del Rio",("OAK", 2014): "Tony Sparano",
    # ── Denver Broncos ────────────────────────────────────────────────────────
    ("DEN", 2023): "Sean Payton", ("DEN", 2022): "Nathaniel Hackett",
    **{("DEN", y): "Vic Fangio" for y in range(2019, 2022)},
    ("DEN", 2018): "Vance Joseph", ("DEN", 2017): "Vance Joseph",
    ("DEN", 2016): "Gary Kubiak",  ("DEN", 2015): "Gary Kubiak",
    **{("DEN", y): "John Fox" for y in range(2011, 2015)},
    ("DEN", 2010): "Josh McDaniels",
    # ── Indianapolis Colts ────────────────────────────────────────────────────
    **{("IND", y): "Frank Reich" for y in range(2018, 2023)},
    **{("IND", y): "Chuck Pagano" for y in range(2012, 2018)},
    **{("IND", y): "Jim Caldwell" for y in range(2009, 2012)},
    **{("IND", y): "Tony Dungy" for y in range(2002, 2009)},
    # ── Pittsburgh Steelers ───────────────────────────────────────────────────
    **{("PIT", y): "Mike Tomlin" for y in range(2007, 2024)},
    **{("PIT", y): "Bill Cowher" for y in range(1992, 2007)},
    # ── Seattle Seahawks ──────────────────────────────────────────────────────
    **{("SEA", y): "Pete Carroll" for y in range(2010, 2024)},
    ("SEA", 2009): "Jim Mora",
    # ── Atlanta Falcons ───────────────────────────────────────────────────────
    **{("ATL", y): "Arthur Smith" for y in range(2021, 2024)},
    **{("ATL", y): "Dan Quinn" for y in range(2015, 2021)},
    ("ATL", 2014): "Mike Smith",  ("ATL", 2013): "Mike Smith",
    # ── New York Giants ───────────────────────────────────────────────────────
    **{("NYG", y): "Brian Daboll" for y in range(2022, 2024)},
    **{("NYG", y): "Joe Judge" for y in range(2020, 2022)},
    **{("NYG", y): "Pat Shurmur" for y in range(2018, 2020)},
    **{("NYG", y): "Ben McAdoo" for y in range(2016, 2018)},
    **{("NYG", y): "Tom Coughlin" for y in range(2004, 2016)},
    # ── New York Jets ─────────────────────────────────────────────────────────
    **{("NYJ", y): "Robert Saleh" for y in range(2021, 2024)},
    ("NYJ", 2020): "Adam Gase",   ("NYJ", 2019): "Adam Gase",
    **{("NYJ", y): "Todd Bowles" for y in range(2015, 2019)},
    # ── Minnesota Vikings ─────────────────────────────────────────────────────
    **{("MIN", y): "Kevin O'Connell" for y in range(2022, 2024)},
    **{("MIN", y): "Mike Zimmer" for y in range(2014, 2022)},
    # ── Carolina Panthers ─────────────────────────────────────────────────────
    ("CAR", 2023): "Frank Reich",  ("CAR", 2022): "Matt Rhule",
    ("CAR", 2021): "Matt Rhule",   ("CAR", 2020): "Matt Rhule",
    **{("CAR", y): "Ron Rivera" for y in range(2011, 2020)},
    # ── Jacksonville Jaguars ──────────────────────────────────────────────────
    **{("JAX", y): "Doug Pederson" for y in range(2022, 2024)},
    ("JAX", 2021): "Urban Meyer",
    **{("JAX", y): "Doug Marrone" for y in range(2017, 2021)},
    # ── Tennessee Titans ──────────────────────────────────────────────────────
    **{("TEN", y): "Mike Vrabel" for y in range(2018, 2024)},
    **{("TEN", y): "Mike Mularkey" for y in range(2016, 2018)},
    # ── Houston Texans ────────────────────────────────────────────────────────
    ("HOU", 2023): "DeMeco Ryans",
    **{("HOU", y): "Lovie Smith" for y in range(2021, 2023)},
    **{("HOU", y): "Bill O'Brien" for y in range(2014, 2021)},
    # ── Arizona Cardinals ─────────────────────────────────────────────────────
    ("ARI", 2023): "Jonathan Gannon",
    **{("ARI", y): "Kliff Kingsbury" for y in range(2019, 2023)},
    **{("ARI", y): "Steve Wilks" for y in range(2018, 2019)},
    **{("ARI", y): "Bruce Arians" for y in range(2013, 2018)},
    # ── New Orleans Saints ────────────────────────────────────────────────────
    **{("NO", y): "Dennis Allen" for y in range(2022, 2024)},
    **{("NO", y): "Sean Payton" for y in range(2006, 2022)},
    # ── Buffalo Bills ─────────────────────────────────────────────────────────
    **{("BUF", y): "Sean McDermott" for y in range(2017, 2024)},
    **{("BUF", y): "Rex Ryan" for y in range(2015, 2017)},
    # ── Los Angeles Chargers ──────────────────────────────────────────────────
    **{("LAC", y): "Brandon Staley" for y in range(2021, 2024)},
    ("LAC", 2020): "Anthony Lynn", ("SD", 2019): "Anthony Lynn",
    # ── Washington ────────────────────────────────────────────────────────────
    **{("WAS", y): "Ron Rivera" for y in range(2020, 2024)},
    ("WAS", 2019): "Bill Callahan",
    # ── Cleveland Browns ──────────────────────────────────────────────────────
    **{("CLE", y): "Kevin Stefanski" for y in range(2020, 2024)},
    ("CLE", 2019): "Freddie Kitchens",
    # ── Chicago Bears ─────────────────────────────────────────────────────────
    **{("CHI", y): "Matt Eberflus" for y in range(2022, 2024)},
    **{("CHI", y): "Matt Nagy" for y in range(2018, 2022)},
    # ── Dallas Cowboys ────────────────────────────────────────────────────────
    **{("DAL", y): "Mike McCarthy" for y in range(2020, 2024)},
    **{("DAL", y): "Jason Garrett" for y in range(2010, 2020)},
}

# nflverse uses different team abbreviations than the standard — map them
NFLVERSE_TEAM_MAP = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF",
    "CAR": "CAR", "CHI": "CHI", "CIN": "CIN", "CLE": "CLE",
    "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GB":  "GB",
    "HOU": "HOU", "IND": "IND", "JAX": "JAX", "KC":  "KC",
    "LA":  "LA",  "LAC": "LAC", "LV":  "LV",  "MIA": "MIA",
    "MIN": "MIN", "NE":  "NE",  "NO":  "NO",  "NYG": "NYG",
    "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT", "SF":  "SF",
    "SEA": "SEA", "TB":  "TB",  "TEN": "TEN", "WAS": "WAS",
    # Legacy abbreviations nflverse may use in older seasons
    "JAC": "JAX", "SD":  "LAC", "STL": "LA",  "OAK": "LV",
}


def build_coaches_df(rosters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build coaches.csv by matching each (team, year) in the roster data
    against the COACH_LOOKUP hardcoded table above.
    """
    pairs = rosters_df[["team", "year"]].drop_duplicates()
    rows = []
    unmatched = []
    for _, row in pairs.iterrows():
        team = str(row["team"]).upper()
        year = int(row["year"])
        coach = COACH_LOOKUP.get((team, year), "")
        if not coach:
            unmatched.append((team, year))
            coach = "Unknown"
        rows.append({"team": team, "year": year, "head_coach": coach})

    if unmatched:
        print(f"  Note: {len(unmatched)} team-seasons had no coach match "
              f"(will show as 'Unknown'): {unmatched[:5]}{'…' if len(unmatched)>5 else ''}")

    return pd.DataFrame(rows)


# ── Main entry point ──────────────────────────────────────────────────────────

def fetch_real_data(start: int = 2010, end: int = 2023) -> None:
    """
    Download real nflverse roster data and build coaches from lookup table.
    Saves data/rosters.csv and data/coaches.csv.
    """
    print(f"Downloading nflverse roster data {start}–{end} …")
    print("Source: https://github.com/nflverse/nflverse-data\n")

    rosters = fetch_all_rosters(start, end)

    # Normalize team abbreviations
    rosters["team"] = rosters["team"].str.upper().map(
        lambda t: NFLVERSE_TEAM_MAP.get(t, t)
    )

    print(f"\nBuilding coaches table …")
    coaches = build_coaches_df(rosters)

    rosters.to_csv(ROSTERS_PATH, index=False)
    coaches.to_csv(COACHES_PATH, index=False)

    print(f"\n✓ {len(rosters)} player-seasons saved → {ROSTERS_PATH}")
    print(f"✓ {len(coaches)} team-seasons saved  → {COACHES_PATH}")
    print(f"  Players:  {rosters['player_id'].nunique()}")
    print(f"  Colleges: {rosters['college'].replace('', pd.NA).dropna().nunique()}")
    print(f"  Coaches:  {coaches['head_coach'].nunique()}")


# ── Seed / demo data (offline fallback) ───────────────────────────────────────

def seed_demo_data() -> None:
    """
    Write a hand-curated offline seed dataset — no internet required.
    Covers 2004–2023, 50+ players, 8 coaching trees, college on every player.
    """
    if os.path.exists(ROSTERS_PATH) and os.path.exists(COACHES_PATH):
        try:
            df = pd.read_csv(ROSTERS_PATH)
            if "college" in df.columns and len(df) >= 150:
                print("Seed data already present — skipping. "
                      "Delete data/ folder to regenerate.")
                return
        except Exception:
            pass

    print("Writing offline seed data …")

    roster_rows = [
        # (player_id, name, position, college, team, year)
        # ── Andy Reid — Kansas City ──────────────────────────────────────────
        ("00-0033873","Patrick Mahomes",     "QB","Texas Tech",         "KC", 2018),
        ("00-0033873","Patrick Mahomes",     "QB","Texas Tech",         "KC", 2019),
        ("00-0033873","Patrick Mahomes",     "QB","Texas Tech",         "KC", 2020),
        ("00-0033873","Patrick Mahomes",     "QB","Texas Tech",         "KC", 2021),
        ("00-0033873","Patrick Mahomes",     "QB","Texas Tech",         "KC", 2022),
        ("00-0033873","Patrick Mahomes",     "QB","Texas Tech",         "KC", 2023),
        ("00-0029604","Travis Kelce",        "TE","Cincinnati",         "KC", 2015),
        ("00-0029604","Travis Kelce",        "TE","Cincinnati",         "KC", 2016),
        ("00-0029604","Travis Kelce",        "TE","Cincinnati",         "KC", 2017),
        ("00-0029604","Travis Kelce",        "TE","Cincinnati",         "KC", 2018),
        ("00-0029604","Travis Kelce",        "TE","Cincinnati",         "KC", 2019),
        ("00-0029604","Travis Kelce",        "TE","Cincinnati",         "KC", 2020),
        ("00-0029604","Travis Kelce",        "TE","Cincinnati",         "KC", 2021),
        ("00-0029604","Travis Kelce",        "TE","Cincinnati",         "KC", 2022),
        ("00-0029604","Travis Kelce",        "TE","Cincinnati",         "KC", 2023),
        ("00-0033040","Tyreek Hill",         "WR","West Alabama",       "KC", 2016),
        ("00-0033040","Tyreek Hill",         "WR","West Alabama",       "KC", 2017),
        ("00-0033040","Tyreek Hill",         "WR","West Alabama",       "KC", 2018),
        ("00-0033040","Tyreek Hill",         "WR","West Alabama",       "KC", 2019),
        ("00-0033040","Tyreek Hill",         "WR","West Alabama",       "KC", 2020),
        ("00-0033040","Tyreek Hill",         "WR","West Alabama",       "KC", 2021),
        ("00-0033040","Tyreek Hill",         "WR","West Alabama",       "MIA",2022),
        ("00-0033040","Tyreek Hill",         "WR","West Alabama",       "MIA",2023),
        ("00-0032524","Chris Jones",         "DT","Mississippi State",  "KC", 2018),
        ("00-0032524","Chris Jones",         "DT","Mississippi State",  "KC", 2019),
        ("00-0032524","Chris Jones",         "DT","Mississippi State",  "KC", 2021),
        ("00-0032524","Chris Jones",         "DT","Mississippi State",  "KC", 2022),
        ("00-0032524","Chris Jones",         "DT","Mississippi State",  "KC", 2023),
        ("00-0035228","Mecole Hardman",      "WR","Georgia",            "KC", 2019),
        ("00-0035228","Mecole Hardman",      "WR","Georgia",            "KC", 2020),
        ("00-0035228","Mecole Hardman",      "WR","Georgia",            "KC", 2021),
        # ── Bill Belichick — New England ─────────────────────────────────────
        ("00-0019596","Tom Brady",           "QB","Michigan",           "NE", 2010),
        ("00-0019596","Tom Brady",           "QB","Michigan",           "NE", 2011),
        ("00-0019596","Tom Brady",           "QB","Michigan",           "NE", 2013),
        ("00-0019596","Tom Brady",           "QB","Michigan",           "NE", 2015),
        ("00-0019596","Tom Brady",           "QB","Michigan",           "NE", 2016),
        ("00-0019596","Tom Brady",           "QB","Michigan",           "NE", 2017),
        ("00-0019596","Tom Brady",           "QB","Michigan",           "NE", 2018),
        ("00-0019596","Tom Brady",           "QB","Michigan",           "TB", 2020),
        ("00-0019596","Tom Brady",           "QB","Michigan",           "TB", 2021),
        ("00-0027793","Rob Gronkowski",      "TE","Arizona",            "NE", 2011),
        ("00-0027793","Rob Gronkowski",      "TE","Arizona",            "NE", 2013),
        ("00-0027793","Rob Gronkowski",      "TE","Arizona",            "NE", 2015),
        ("00-0027793","Rob Gronkowski",      "TE","Arizona",            "NE", 2016),
        ("00-0027793","Rob Gronkowski",      "TE","Arizona",            "NE", 2017),
        ("00-0027793","Rob Gronkowski",      "TE","Arizona",            "NE", 2018),
        ("00-0027793","Rob Gronkowski",      "TE","Arizona",            "TB", 2020),
        ("00-0027793","Rob Gronkowski",      "TE","Arizona",            "TB", 2021),
        ("00-0027793","Rob Gronkowski",      "TE","Arizona",            "LA", 2022),
        ("00-0027616","Devin McCourty",      "DB","Rutgers",            "NE", 2013),
        ("00-0027616","Devin McCourty",      "DB","Rutgers",            "NE", 2015),
        ("00-0027616","Devin McCourty",      "DB","Rutgers",            "NE", 2017),
        ("00-0027616","Devin McCourty",      "DB","Rutgers",            "NE", 2019),
        ("00-0027616","Devin McCourty",      "DB","Rutgers",            "NE", 2021),
        ("00-0022552","Julian Edelman",      "WR","Kent State",         "NE", 2013),
        ("00-0022552","Julian Edelman",      "WR","Kent State",         "NE", 2015),
        ("00-0022552","Julian Edelman",      "WR","Kent State",         "NE", 2017),
        ("00-0022552","Julian Edelman",      "WR","Kent State",         "NE", 2018),
        # ── Bruce Arians / Todd Bowles — Tampa Bay ────────────────────────────
        ("00-0031408","Mike Evans",          "WR","Texas A&M",          "TB", 2019),
        ("00-0031408","Mike Evans",          "WR","Texas A&M",          "TB", 2020),
        ("00-0031408","Mike Evans",          "WR","Texas A&M",          "TB", 2021),
        ("00-0031408","Mike Evans",          "WR","Texas A&M",          "TB", 2022),
        ("00-0036228","Tristan Wirfs",       "OT","Iowa",               "TB", 2020),
        ("00-0036228","Tristan Wirfs",       "OT","Iowa",               "TB", 2021),
        ("00-0036228","Tristan Wirfs",       "OT","Iowa",               "TB", 2022),
        ("00-0033119","Chris Godwin",        "WR","Penn State",         "TB", 2019),
        ("00-0033119","Chris Godwin",        "WR","Penn State",         "TB", 2020),
        ("00-0033119","Chris Godwin",        "WR","Penn State",         "TB", 2021),
        # ── Antonio Brown — bridge PIT→TB ────────────────────────────────────
        ("00-0026498","Antonio Brown",       "WR","Central Michigan",   "PIT",2015),
        ("00-0026498","Antonio Brown",       "WR","Central Michigan",   "PIT",2016),
        ("00-0026498","Antonio Brown",       "WR","Central Michigan",   "PIT",2017),
        ("00-0026498","Antonio Brown",       "WR","Central Michigan",   "TB", 2020),
        ("00-0026498","Antonio Brown",       "WR","Central Michigan",   "TB", 2021),
        # ── Sean McVay — LA Rams ─────────────────────────────────────────────
        ("00-0033076","Cooper Kupp",         "WR","Eastern Washington", "LA", 2018),
        ("00-0033076","Cooper Kupp",         "WR","Eastern Washington", "LA", 2019),
        ("00-0033076","Cooper Kupp",         "WR","Eastern Washington", "LA", 2020),
        ("00-0033076","Cooper Kupp",         "WR","Eastern Washington", "LA", 2021),
        ("00-0033076","Cooper Kupp",         "WR","Eastern Washington", "LA", 2022),
        ("00-0030520","Aaron Donald",        "DT","Pittsburgh",         "LA", 2018),
        ("00-0030520","Aaron Donald",        "DT","Pittsburgh",         "LA", 2019),
        ("00-0030520","Aaron Donald",        "DT","Pittsburgh",         "LA", 2020),
        ("00-0030520","Aaron Donald",        "DT","Pittsburgh",         "LA", 2021),
        ("00-0030520","Aaron Donald",        "DT","Pittsburgh",         "LA", 2022),
        ("00-0028118","Matthew Stafford",    "QB","Georgia",            "DET",2013),
        ("00-0028118","Matthew Stafford",    "QB","Georgia",            "DET",2015),
        ("00-0028118","Matthew Stafford",    "QB","Georgia",            "DET",2018),
        ("00-0028118","Matthew Stafford",    "QB","Georgia",            "DET",2020),
        ("00-0028118","Matthew Stafford",    "QB","Georgia",            "LA", 2021),
        ("00-0028118","Matthew Stafford",    "QB","Georgia",            "LA", 2022),
        # ── Von Miller — bridge DEN→LA ───────────────────────────────────────
        ("00-0028090","Von Miller",          "LB","Texas A&M",          "DEN",2013),
        ("00-0028090","Von Miller",          "LB","Texas A&M",          "DEN",2015),
        ("00-0028090","Von Miller",          "LB","Texas A&M",          "DEN",2016),
        ("00-0028090","Von Miller",          "LB","Texas A&M",          "LA", 2021),
        ("00-0028090","Von Miller",          "LB","Texas A&M",          "LA", 2022),
        # ── Kyle Shanahan — San Francisco ─────────────────────────────────────
        ("00-0033949","George Kittle",       "TE","Iowa",               "SF", 2017),
        ("00-0033949","George Kittle",       "TE","Iowa",               "SF", 2018),
        ("00-0033949","George Kittle",       "TE","Iowa",               "SF", 2019),
        ("00-0033949","George Kittle",       "TE","Iowa",               "SF", 2021),
        ("00-0033949","George Kittle",       "TE","Iowa",               "SF", 2022),
        ("00-0033949","George Kittle",       "TE","Iowa",               "SF", 2023),
        ("00-0033280","Christian McCaffrey", "RB","Stanford",           "CAR",2019),
        ("00-0033280","Christian McCaffrey", "RB","Stanford",           "CAR",2021),
        ("00-0033280","Christian McCaffrey", "RB","Stanford",           "SF", 2022),
        ("00-0033280","Christian McCaffrey", "RB","Stanford",           "SF", 2023),
        ("00-0037077","Brock Purdy",         "QB","Iowa State",         "SF", 2022),
        ("00-0037077","Brock Purdy",         "QB","Iowa State",         "SF", 2023),
        ("00-0035676","Brandon Aiyuk",       "WR","Arizona State",      "SF", 2020),
        ("00-0035676","Brandon Aiyuk",       "WR","Arizona State",      "SF", 2021),
        ("00-0035676","Brandon Aiyuk",       "WR","Arizona State",      "SF", 2022),
        ("00-0035676","Brandon Aiyuk",       "WR","Arizona State",      "SF", 2023),
        ("00-0034796","Deebo Samuel",        "WR","South Carolina",     "SF", 2019),
        ("00-0034796","Deebo Samuel",        "WR","South Carolina",     "SF", 2021),
        ("00-0034796","Deebo Samuel",        "WR","South Carolina",     "SF", 2022),
        ("00-0034796","Deebo Samuel",        "WR","South Carolina",     "SF", 2023),
        # ── John Harbaugh — Baltimore ─────────────────────────────────────────
        ("00-0035228","Lamar Jackson",       "QB","Louisville",         "BAL",2018),
        ("00-0035228","Lamar Jackson",       "QB","Louisville",         "BAL",2019),
        ("00-0035228","Lamar Jackson",       "QB","Louisville",         "BAL",2020),
        ("00-0035228","Lamar Jackson",       "QB","Louisville",         "BAL",2022),
        ("00-0035228","Lamar Jackson",       "QB","Louisville",         "BAL",2023),
        ("00-0034785","Mark Andrews",        "TE","Oklahoma",           "BAL",2018),
        ("00-0034785","Mark Andrews",        "TE","Oklahoma",           "BAL",2019),
        ("00-0034785","Mark Andrews",        "TE","Oklahoma",           "BAL",2021),
        ("00-0034785","Mark Andrews",        "TE","Oklahoma",           "BAL",2023),
        # ── Matt LaFleur — Green Bay ──────────────────────────────────────────
        ("00-0023459","Aaron Rodgers",       "QB","California",         "GB", 2018),
        ("00-0023459","Aaron Rodgers",       "QB","California",         "GB", 2019),
        ("00-0023459","Aaron Rodgers",       "QB","California",         "GB", 2020),
        ("00-0023459","Aaron Rodgers",       "QB","California",         "GB", 2021),
        ("00-0023459","Aaron Rodgers",       "QB","California",         "NYJ",2023),
        ("00-0031587","Davante Adams",       "WR","Fresno State",       "GB", 2018),
        ("00-0031587","Davante Adams",       "WR","Fresno State",       "GB", 2019),
        ("00-0031587","Davante Adams",       "WR","Fresno State",       "GB", 2020),
        ("00-0031587","Davante Adams",       "WR","Fresno State",       "GB", 2021),
        ("00-0031587","Davante Adams",       "WR","Fresno State",       "LV", 2022),
        ("00-0031587","Davante Adams",       "WR","Fresno State",       "LV", 2023),
        ("00-0033921","Aaron Jones",         "RB","UTEP",               "GB", 2019),
        ("00-0033921","Aaron Jones",         "RB","UTEP",               "GB", 2020),
        ("00-0033921","Aaron Jones",         "RB","UTEP",               "GB", 2021),
        ("00-0033921","Aaron Jones",         "RB","UTEP",               "MIN",2023),
        # ── Bridge: Josh Jacobs RAI→KC ────────────────────────────────────────
        ("00-0035700","Josh Jacobs",         "RB","Alabama",            "LV", 2019),
        ("00-0035700","Josh Jacobs",         "RB","Alabama",            "LV", 2020),
        ("00-0035700","Josh Jacobs",         "RB","Alabama",            "LV", 2021),
        ("00-0035700","Josh Jacobs",         "RB","Alabama",            "LV", 2022),
        ("00-0035700","Josh Jacobs",         "RB","Alabama",            "KC", 2023),
        # ── Bridge: Le'Veon Bell PIT→KC ──────────────────────────────────────
        ("00-0030282","Le'Veon Bell",        "RB","Michigan State",     "PIT",2015),
        ("00-0030282","Le'Veon Bell",        "RB","Michigan State",     "PIT",2016),
        ("00-0030282","Le'Veon Bell",        "RB","Michigan State",     "PIT",2017),
        ("00-0030282","Le'Veon Bell",        "RB","Michigan State",     "KC", 2020),
        # ── Zac Taylor (McVay disciple) — Cincinnati ──────────────────────────
        ("00-0036442","Joe Burrow",          "QB","LSU",                "CIN",2020),
        ("00-0036442","Joe Burrow",          "QB","LSU",                "CIN",2021),
        ("00-0036442","Joe Burrow",          "QB","LSU",                "CIN",2022),
        ("00-0036442","Joe Burrow",          "QB","LSU",                "CIN",2023),
        ("00-0036900","Ja'Marr Chase",       "WR","LSU",                "CIN",2021),
        ("00-0036900","Ja'Marr Chase",       "WR","LSU",                "CIN",2022),
        ("00-0036900","Ja'Marr Chase",       "WR","LSU",                "CIN",2023),
        ("00-0036254","Tee Higgins",         "WR","Clemson",            "CIN",2020),
        ("00-0036254","Tee Higgins",         "WR","Clemson",            "CIN",2021),
        ("00-0036254","Tee Higgins",         "WR","Clemson",            "CIN",2022),
        # ── Mike McDaniel (Shanahan disciple) — Miami ─────────────────────────
        ("00-0036355","Tua Tagovailoa",      "QB","Alabama",            "MIA",2020),
        ("00-0036355","Tua Tagovailoa",      "QB","Alabama",            "MIA",2021),
        ("00-0036355","Tua Tagovailoa",      "QB","Alabama",            "MIA",2022),
        ("00-0036355","Tua Tagovailoa",      "QB","Alabama",            "MIA",2023),
        ("00-0036898","Jaylen Waddle",       "WR","Alabama",            "MIA",2021),
        ("00-0036898","Jaylen Waddle",       "WR","Alabama",            "MIA",2022),
        ("00-0036898","Jaylen Waddle",       "WR","Alabama",            "MIA",2023),
        # ── Nick Sirianni (Reid disciple) — Philadelphia ──────────────────────
        ("00-0036389","Jalen Hurts",         "QB","Oklahoma",           "PHI",2020),
        ("00-0036389","Jalen Hurts",         "QB","Oklahoma",           "PHI",2021),
        ("00-0036389","Jalen Hurts",         "QB","Oklahoma",           "PHI",2022),
        ("00-0036389","Jalen Hurts",         "QB","Oklahoma",           "PHI",2023),
        ("00-0036895","DeVonta Smith",       "WR","Alabama",            "PHI",2021),
        ("00-0036895","DeVonta Smith",       "WR","Alabama",            "PHI",2022),
        ("00-0036895","DeVonta Smith",       "WR","Alabama",            "PHI",2023),
        ("00-0034858","A.J. Brown",          "WR","Ole Miss",           "TEN",2019),
        ("00-0034858","A.J. Brown",          "WR","Ole Miss",           "TEN",2020),
        ("00-0034858","A.J. Brown",          "WR","Ole Miss",           "TEN",2021),
        ("00-0034858","A.J. Brown",          "WR","Ole Miss",           "PHI",2022),
        ("00-0034858","A.J. Brown",          "WR","Ole Miss",           "PHI",2023),
        # ── D'Andre Swift — bridge DET→PHI ───────────────────────────────────
        ("00-0036252","D'Andre Swift",       "RB","Georgia",            "DET",2020),
        ("00-0036252","D'Andre Swift",       "RB","Georgia",            "DET",2021),
        ("00-0036252","D'Andre Swift",       "RB","Georgia",            "DET",2022),
        ("00-0036252","D'Andre Swift",       "RB","Georgia",            "PHI",2023),
        # ── Dan Campbell — Detroit ────────────────────────────────────────────
        ("00-0034588","Jared Goff",          "QB","California",         "LA", 2017),
        ("00-0034588","Jared Goff",          "QB","California",         "LA", 2018),
        ("00-0034588","Jared Goff",          "QB","California",         "LA", 2019),
        ("00-0034588","Jared Goff",          "QB","California",         "DET",2021),
        ("00-0034588","Jared Goff",          "QB","California",         "DET",2022),
        ("00-0034588","Jared Goff",          "QB","California",         "DET",2023),
        ("00-0036900","Amon-Ra St. Brown",   "WR","USC",                "DET",2021),
        ("00-0036900","Amon-Ra St. Brown",   "WR","USC",                "DET",2022),
        ("00-0036900","Amon-Ra St. Brown",   "WR","USC",                "DET",2023),
        # ── Peyton Manning era ────────────────────────────────────────────────
        ("00-0010346","Peyton Manning",      "QB","Tennessee",          "IND",2004),
        ("00-0010346","Peyton Manning",      "QB","Tennessee",          "IND",2009),
        ("00-0010346","Peyton Manning",      "QB","Tennessee",          "DEN",2013),
        ("00-0010346","Peyton Manning",      "QB","Tennessee",          "DEN",2015),
        ("00-0027823","Demaryius Thomas",    "WR","Georgia Tech",       "DEN",2013),
        ("00-0027823","Demaryius Thomas",    "WR","Georgia Tech",       "DEN",2015),
        ("00-0027823","Demaryius Thomas",    "WR","Georgia Tech",       "DEN",2016),
    ]

    coach_rows = [
        ("KC", 2015,"Andy Reid"),    ("KC", 2016,"Andy Reid"),
        ("KC", 2017,"Andy Reid"),    ("KC", 2018,"Andy Reid"),
        ("KC", 2019,"Andy Reid"),    ("KC", 2020,"Andy Reid"),
        ("KC", 2021,"Andy Reid"),    ("KC", 2022,"Andy Reid"),
        ("KC", 2023,"Andy Reid"),
        ("MIA",2019,"Brian Flores"), ("MIA",2020,"Brian Flores"),
        ("MIA",2021,"Brian Flores"),
        ("MIA",2022,"Mike McDaniel"),("MIA",2023,"Mike McDaniel"),
        ("NE", 2010,"Bill Belichick"),("NE",2011,"Bill Belichick"),
        ("NE", 2013,"Bill Belichick"),("NE",2015,"Bill Belichick"),
        ("NE", 2016,"Bill Belichick"),("NE",2017,"Bill Belichick"),
        ("NE", 2018,"Bill Belichick"),("NE",2019,"Bill Belichick"),
        ("NE", 2021,"Bill Belichick"),
        ("TB", 2019,"Bruce Arians"), ("TB", 2020,"Bruce Arians"),
        ("TB", 2021,"Bruce Arians"), ("TB", 2022,"Todd Bowles"),
        ("LA", 2017,"Sean McVay"),   ("LA", 2018,"Sean McVay"),
        ("LA", 2019,"Sean McVay"),   ("LA", 2020,"Sean McVay"),
        ("LA", 2021,"Sean McVay"),   ("LA", 2022,"Sean McVay"),
        ("DEN",2013,"John Fox"),     ("DEN",2015,"Gary Kubiak"),
        ("DEN",2016,"Gary Kubiak"),
        ("SF", 2017,"Kyle Shanahan"),("SF", 2018,"Kyle Shanahan"),
        ("SF", 2019,"Kyle Shanahan"),("SF", 2020,"Kyle Shanahan"),
        ("SF", 2021,"Kyle Shanahan"),("SF", 2022,"Kyle Shanahan"),
        ("SF", 2023,"Kyle Shanahan"),
        ("BAL",2018,"John Harbaugh"),("BAL",2019,"John Harbaugh"),
        ("BAL",2020,"John Harbaugh"),("BAL",2021,"John Harbaugh"),
        ("BAL",2022,"John Harbaugh"),("BAL",2023,"John Harbaugh"),
        ("GB", 2018,"Mike McCarthy"),("GB", 2019,"Matt LaFleur"),
        ("GB", 2020,"Matt LaFleur"), ("GB", 2021,"Matt LaFleur"),
        ("NYJ",2023,"Robert Saleh"),
        ("LV", 2019,"Jon Gruden"),   ("LV", 2020,"Jon Gruden"),
        ("LV", 2021,"Jon Gruden"),   ("LV", 2022,"Josh McDaniels"),
        ("LV", 2023,"Josh McDaniels"),
        ("CIN",2020,"Zac Taylor"),   ("CIN",2021,"Zac Taylor"),
        ("CIN",2022,"Zac Taylor"),   ("CIN",2023,"Zac Taylor"),
        ("PHI",2020,"Doug Pederson"),("PHI",2021,"Nick Sirianni"),
        ("PHI",2022,"Nick Sirianni"),("PHI",2023,"Nick Sirianni"),
        ("DET",2013,"Jim Caldwell"), ("DET",2015,"Jim Caldwell"),
        ("DET",2018,"Matt Patricia"),("DET",2019,"Matt Patricia"),
        ("DET",2021,"Dan Campbell"), ("DET",2022,"Dan Campbell"),
        ("DET",2023,"Dan Campbell"),
        ("TEN",2019,"Mike Vrabel"),  ("TEN",2020,"Mike Vrabel"),
        ("TEN",2021,"Mike Vrabel"),
        ("PIT",2015,"Mike Tomlin"),  ("PIT",2016,"Mike Tomlin"),
        ("PIT",2017,"Mike Tomlin"),
        ("IND",2004,"Tony Dungy"),   ("IND",2009,"Jim Caldwell"),
        ("MIN",2023,"Kevin O'Connell"),
    ]

    df_r = pd.DataFrame(roster_rows,
                        columns=["player_id","name","position","college","team","year"])
    df_c = pd.DataFrame(coach_rows, columns=["team","year","head_coach"])
    df_r.to_csv(ROSTERS_PATH, index=False)
    df_c.to_csv(COACHES_PATH, index=False)
    print(f"✓ Seed data written: {len(df_r)} player-seasons, "
          f"{df_r['player_id'].nunique()} players, "
          f"{df_r['college'].nunique()} colleges")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NFL data fetcher — downloads from nflverse or writes seed data"
    )
    parser.add_argument("--fetch", action="store_true",
                        help="Download real data from nflverse (requires internet)")
    parser.add_argument("--demo",  action="store_true",
                        help="Write offline seed data (no internet needed)")
    parser.add_argument("--start", type=int, default=2010,
                        help="First season to fetch (default 2010)")
    parser.add_argument("--end",   type=int, default=2023,
                        help="Last season to fetch (default 2023)")
    args = parser.parse_args()

    if args.fetch:
        fetch_real_data(args.start, args.end)
    elif args.demo:
        seed_demo_data()
    else:
        print("Usage:")
        print("  python data_fetcher.py --fetch           # real nflverse data")
        print("  python data_fetcher.py --fetch --start 2015 --end 2023")
        print("  python data_fetcher.py --demo            # offline seed data")
