# NFL Coaching Network Explorer
### SI 507 Final Project

A graph-based application that maps NFL players through their coaching networks, revealing how players, coaches, and systems are interconnected across teams and eras.

---

## Unique Angle: Coaching Trees
Unlike a simple "teammates graph," this project uses a bipartite graph connecting players to coaches. This reveals:
- Which coaches share the most players (the Belichick tree, the Reid disciples)
- How players move between coaching systems
- Shortest connection between any two players through shared coaches

---

## Quick Start

# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch real NFLverse data (~30 seconds)
python3 data_fetcher.py --fetch --start 2010 --end 2023

# 3. Launch the Streamlit app
streamlit run app.py

# Optional: use offline seed data instead of fetching
python3 data_fetcher.py --demo

```

---

## Project Structure

```
nfl_network/
├── app.py              # Streamlit UI — all 5 interaction modes
├── nfl_graph.py        # Core graph module (Player, Coach, NFLGraph classes)
├── data_fetcher.py     # PFR scraper + demo data seeder
├── requirements.txt
├── data/
│   ├── rosters.csv     # player_id, name, position, team, year
│   └── coaches.csv     # team, year, head_coach
└── tests/
    └── test_nfl_graph.py   # 40+ pytest tests
```

---

## Graph Structure

| Element | Description |
|---------|-------------|
| **Player node** | Each unique player; attrs: name, position, teams, seasons |
| **Coach node**  | Each head coach; attrs: name, teams, seasons coached |
| **Edge**        | Player ↔ Coach; weight = seasons played together |

---

## Interaction Modes

| # | Mode | Description |
|---|------|-------------|
| 1 | **Player Profile** | Search any player → bio, coaching history, network viz |
| 2 | **Teammates via Coaches** | Players connected through shared coaching systems |
| 3 | **Shortest Path** | "Six Degrees of NFL" — find the path between any two players |
| 4 | **Centrality Rankings** | Most connected players by degree centrality |
| 5 | **Coaching Tree** 🌳 | Full network view around a head coach — players + coaching family |

---

## Data Sources

-**Roster Data:** [nflverse](https://github.com/nflverse/nflverse-data)  
Open-source NFL roster CSVs maintained by the nflverse team. Provides player 
name, position, team, college, and NFL player ID for every season going back 
to 1999. Downloaded via `data_fetcher.py --fetch`.

**Coaching Data:** Hardcoded lookup table in `data_fetcher.py`  
Head coach per team per season (2000–2023) for all 32 NFL teams, compiled 
from public NFL records.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

40+ tests covering:
- Data loading & CSV validation
- Player/Coach dataclass correctness
- Graph construction (nodes, edges, weights)
- All 5 interaction modes
- Edge cases and error handling

---

## Key Classes

### `Player` (dataclass)
```python
Player(player_id, name, positions, teams, seasons, coaches)
# Properties: primary_position, career_span, num_seasons
```

### `Coach` (dataclass)
```python
Coach(name, teams, seasons, players)
# Properties: career_span, num_seasons, teams_coached
```

### `NFLGraph`
```python
g = NFLGraph()
g.build("data/rosters.csv", "data/coaches.csv")

g.find_player("Brady")               # search
g.player_profile(pid)                 # Mode 1
g.get_teammates(pid)                  # Mode 2
g.shortest_path(pid1, pid2)           # Mode 3
g.centrality_rankings(top_n=20)       # Mode 4
g.coaching_tree("Bill Belichick")     # Mode 5
g.player_projection()                 # NetworkX graph (players only)
```
