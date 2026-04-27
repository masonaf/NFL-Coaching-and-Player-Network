"""
nfl_graph.py
------------
Core module for the NFL Player Network project.

Classes
-------
Player       — represents an NFL player node
Coach        — represents an NFL coach node
NFLGraph     — builds and queries the bipartite player-coach network

The network has two types of nodes:
  • Player nodes  (type="player")
  • Coach nodes   (type="coach")

Edges connect a player to every coach they played under.
Edge weight = number of seasons together.

This "coaching tree" graph reveals:
  • Which coaches share the most players (coaching families / disciples)
  • How players bounce between coaching systems
  • Shortest path between any two players through shared coaches
  • Most influential coaches by player connectivity
"""

import os
import pandas as pd
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Player:
    """Represents one NFL player."""
    player_id: str
    name: str
    positions: list[str] = field(default_factory=list)
    teams: list[str]     = field(default_factory=list)
    seasons: list[int]   = field(default_factory=list)
    coaches: list[str]   = field(default_factory=list)   # coach names
    college: str         = ""

    @property
    def primary_position(self) -> str:
        if not self.positions:
            return "UNK"
        # Return most common position
        return max(set(self.positions), key=self.positions.count)

    @property
    def career_span(self) -> str:
        if not self.seasons:
            return "Unknown"
        return f"{min(self.seasons)}–{max(self.seasons)}"

    @property
    def num_seasons(self) -> int:
        return len(set(self.seasons))

    def __repr__(self) -> str:
        return (f"Player({self.name!r}, {self.primary_position}, "
                f"{self.career_span}, college={self.college!r}, teams={list(set(self.teams))})")


@dataclass
class Coach:
    """Represents an NFL head coach."""
    name: str
    teams: list[str]   = field(default_factory=list)
    seasons: list[int] = field(default_factory=list)
    players: list[str] = field(default_factory=list)   # player_ids

    @property
    def career_span(self) -> str:
        if not self.seasons:
            return "Unknown"
        return f"{min(self.seasons)}–{max(self.seasons)}"

    @property
    def num_seasons(self) -> int:
        return len(set(self.seasons))

    @property
    def teams_coached(self) -> list[str]:
        return list(set(self.teams))

    def __repr__(self) -> str:
        return (f"Coach({self.name!r}, {self.career_span}, "
                f"teams={self.teams_coached})")


# ── Main graph class ──────────────────────────────────────────────────────────

class NFLGraph:
    """
    Bipartite graph connecting NFL players to the coaches they played under.

    Nodes
    -----
    Player node : id = player_id,  attr type="player"
    Coach node  : id = "COACH::<name>", attr type="coach"

    Edges
    -----
    (player_id, "COACH::<coach>")  weight = seasons_together
    """

    COACH_PREFIX = "COACH::"

    def __init__(self):
        self.G: nx.Graph = nx.Graph()
        self.players: dict[str, Player] = {}   # player_id → Player
        self.coaches: dict[str, Coach]  = {}   # name → Coach
        self._built = False

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self, rosters_path: str, coaches_path: str) -> None:
        """
        Load CSVs and construct the player-coach bipartite graph.

        Parameters
        ----------
        rosters_path : str   path to data/rosters.csv
        coaches_path : str   path to data/coaches.csv
        """
        df_r = pd.read_csv(rosters_path)
        df_c = pd.read_csv(coaches_path)

        # Merge to get coach per player-team-season
        df = df_r.merge(df_c, on=["team", "year"], how="left")
        df["head_coach"] = df["head_coach"].fillna("Unknown")

        # ── Build Player objects ──────────────────────────────────────────
        for pid, grp in df.groupby("player_id"):
            college = ""
            if "college" in grp.columns:
                colleges = grp["college"].dropna().unique()
                college = colleges[0] if len(colleges) > 0 else ""
            p = Player(
                player_id=pid,
                name=grp["name"].iloc[0],
                positions=grp["position"].tolist(),
                teams=grp["team"].tolist(),
                seasons=grp["year"].tolist(),
                coaches=grp["head_coach"].tolist(),
                college=college,
            )
            self.players[pid] = p

        # ── Build Coach objects ───────────────────────────────────────────
        for coach_name, grp in df.groupby("head_coach"):
            c = Coach(
                name=coach_name,
                teams=grp["team"].tolist(),
                seasons=grp["year"].tolist(),
                players=grp["player_id"].tolist(),
            )
            self.coaches[coach_name] = c

        # ── Add nodes ─────────────────────────────────────────────────────
        for pid, p in self.players.items():
            self.G.add_node(pid,
                            label=p.name,
                            type="player",
                            position=p.primary_position,
                            college=p.college,
                            teams=list(set(p.teams)),
                            seasons=list(set(p.seasons)))

        for cname, c in self.coaches.items():
            node_id = self.COACH_PREFIX + cname
            self.G.add_node(node_id,
                            label=cname,
                            type="coach",
                            teams=list(set(c.teams)),
                            seasons=list(set(c.seasons)))

        # ── Add edges (player ↔ coach, weight = seasons together) ─────────
        # Count seasons per (player_id, coach) pair
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        for _, row in df.iterrows():
            pid   = row["player_id"]
            coach = row["head_coach"]
            pair_counts[(pid, coach)] += 1

        for (pid, coach), seasons_together in pair_counts.items():
            coach_node = self.COACH_PREFIX + coach
            self.G.add_edge(pid, coach_node,
                            weight=seasons_together,
                            seasons_together=seasons_together)

        self._built = True
        print(f"✓ Graph built: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")

    # ── Lookup helpers ────────────────────────────────────────────────────────

    def _require_built(self):
        if not self._built:
            raise RuntimeError("Call build() before querying the graph.")

    def find_player(self, query: str) -> list[Player]:
        """Return players whose name contains `query` (case-insensitive)."""
        self._require_built()
        q = query.strip().lower()
        return [p for p in self.players.values()
                if q in p.name.lower()]

    def find_coach(self, query: str) -> list[Coach]:
        """Return coaches whose name contains `query` (case-insensitive)."""
        self._require_built()
        q = query.strip().lower()
        return [c for c in self.coaches.values()
                if q in c.name.lower()]

    # ── Mode 1: Player profile ────────────────────────────────────────────────

    def player_profile(self, player_id: str) -> dict:
        """
        Return full profile for a player: bio, teammates, coaches.
        """
        self._require_built()
        if player_id not in self.players:
            return {}
        p = self.players[player_id]

        # Coaches this player played under
        coach_nodes = list(self.G.neighbors(player_id))
        coaches_info = []
        for cn in coach_nodes:
            cname = cn.replace(self.COACH_PREFIX, "")
            w = self.G[player_id][cn]["seasons_together"]
            coaches_info.append({"coach": cname, "seasons_together": w})
        coaches_info.sort(key=lambda x: -x["seasons_together"])

        # Teammates = other players connected to same coaches
        teammates: dict[str, int] = defaultdict(int)
        for cn in coach_nodes:
            for neighbor in self.G.neighbors(cn):
                if neighbor != player_id and not neighbor.startswith(self.COACH_PREFIX):
                    teammates[neighbor] += 1
        top_teammates = sorted(teammates.items(), key=lambda x: -x[1])[:10]
        teammate_objs = [
            {"player": self.players[pid].name,
             "player_id": pid,
             "shared_coaches": cnt}
            for pid, cnt in top_teammates
            if pid in self.players
        ]

        return {
            "player":    p,
            "coaches":   coaches_info,
            "teammates": teammate_objs,
        }

    # ── Mode 2: Teammates of a player ─────────────────────────────────────────

    def get_teammates(self, player_id: str, min_shared_coaches: int = 1) -> list[dict]:
        """
        Return all players connected through at least one shared coach.
        """
        self._require_built()
        teammates: dict[str, int] = defaultdict(int)
        for cn in self.G.neighbors(player_id):
            if not cn.startswith(self.COACH_PREFIX):
                continue
            for neighbor in self.G.neighbors(cn):
                if neighbor != player_id and not neighbor.startswith(self.COACH_PREFIX):
                    teammates[neighbor] += 1

        result = []
        for pid, shared in teammates.items():
            if shared >= min_shared_coaches and pid in self.players:
                p = self.players[pid]
                result.append({
                    "player_id": pid,
                    "name": p.name,
                    "position": p.primary_position,
                    "shared_coaches": shared,
                    "teams": list(set(p.teams)),
                })
        return sorted(result, key=lambda x: -x["shared_coaches"])

    # ── Mode 3: Shortest path between two players ─────────────────────────────

    def shortest_path(self, pid1: str, pid2: str) -> dict:
        """
        Find the shortest connection between two players through
        the coaching network.

        Returns path with human-readable labels and connection explanation.
        """
        self._require_built()
        try:
            path = nx.shortest_path(self.G, pid1, pid2)
        except nx.NetworkXNoPath:
            return {"found": False, "reason": "No path exists between these players."}
        except nx.NodeNotFound as e:
            return {"found": False, "reason": str(e)}

        # Build human-readable steps
        steps = []
        for node in path:
            if node.startswith(self.COACH_PREFIX):
                cname = node.replace(self.COACH_PREFIX, "")
                steps.append({"type": "coach", "label": cname, "id": node})
            else:
                p = self.players.get(node)
                label = p.name if p else node
                steps.append({"type": "player", "label": label, "id": node})

        degree_of_separation = (len(path) - 1) // 2  # each hop = player→coach→player

        return {
            "found": True,
            "path": path,
            "steps": steps,
            "degrees": degree_of_separation,
            "length": len(path) - 1,
        }

    # ── Mode 4: Top players by centrality ────────────────────────────────────

    def centrality_rankings(self, top_n: int = 20) -> list[dict]:
        """
        Rank players by degree centrality in the coaching network.
        High centrality = connected to many coaches = journeyman / system linchpin.
        """
        self._require_built()
        centrality = nx.degree_centrality(self.G)
        player_centrality = {
            pid: centrality[pid]
            for pid in self.players
            if pid in centrality
        }
        ranked = sorted(player_centrality.items(), key=lambda x: -x[1])[:top_n]
        results = []
        for pid, score in ranked:
            p = self.players[pid]
            results.append({
                "player_id": pid,
                "name": p.name,
                "position": p.primary_position,
                "centrality": round(score, 4),
                "num_coaches": self.G.degree(pid),
                "teams": list(set(p.teams)),
                "career_span": p.career_span,
            })
        return results

    # ── Mode 5: Coach tree (bonus / A+ differentiator) ───────────────────────

    def coaching_tree(self, coach_name: str) -> dict:
        """
        Show the full network around a head coach:
        - Players they coached
        - Other coaches those players also played for ("coaching family")
        - Sorted by shared players
        """
        self._require_built()
        coach_node = self.COACH_PREFIX + coach_name
        if coach_node not in self.G:
            return {}

        coach = self.coaches.get(coach_name)
        player_nodes = [n for n in self.G.neighbors(coach_node)
                        if not n.startswith(self.COACH_PREFIX)]

        # Related coaches — coaches that share players with this coach
        related: dict[str, int] = defaultdict(int)
        for pid in player_nodes:
            for cn in self.G.neighbors(pid):
                if cn.startswith(self.COACH_PREFIX) and cn != coach_node:
                    related[cn] += 1

        related_coaches = []
        for cn, shared in sorted(related.items(), key=lambda x: -x[1]):
            cname = cn.replace(self.COACH_PREFIX, "")
            related_coaches.append({
                "coach": cname,
                "shared_players": shared,
            })

        players_info = []
        for pid in player_nodes:
            p = self.players.get(pid)
            if p:
                seasons_together = self.G[pid][coach_node].get("seasons_together", 0)
                players_info.append({
                    "player_id": pid,
                    "name": p.name,
                    "position": p.primary_position,
                    "seasons_together": seasons_together,
                })
        players_info.sort(key=lambda x: -x["seasons_together"])

        return {
            "coach": coach,
            "players": players_info,
            "related_coaches": related_coaches[:15],
            "total_players": len(players_info),
        }

    # ── College methods (professor-suggested feature) ─────────────────────────

    def college_alumni(self, college_name: str) -> list[dict]:
        """
        Return all players from a given college (case-insensitive partial match).
        Useful for seeing which schools produce the most NFL talent
        and how those players are spread across coaching systems.
        """
        self._require_built()
        q = college_name.strip().lower()
        results = []
        for pid, p in self.players.items():
            if q in p.college.lower():
                results.append({
                    "player_id": pid,
                    "name": p.name,
                    "position": p.primary_position,
                    "college": p.college,
                    "teams": list(set(p.teams)),
                    "career_span": p.career_span,
                    "coaches": list(set(p.coaches)),
                })
        return sorted(results, key=lambda x: x["name"])

    def college_pipeline(self, top_n: int = 20) -> list[dict]:
        """
        Rank colleges by number of NFL players in the dataset.
        Reveals which schools are most connected to the league's
        coaching networks (e.g., Alabama, Georgia, Ohio State pipelines).
        """
        self._require_built()
        from collections import Counter
        college_counts: Counter = Counter()
        college_players: dict[str, list[str]] = defaultdict(list)
        college_teams: dict[str, set] = defaultdict(set)

        for p in self.players.values():
            if p.college:
                college_counts[p.college] += 1
                college_players[p.college].append(p.name)
                for t in p.teams:
                    college_teams[p.college].add(t)

        results = []
        for college, count in college_counts.most_common(top_n):
            results.append({
                "college": college,
                "player_count": count,
                "nfl_teams_reached": len(college_teams[college]),
                "players": sorted(college_players[college]),
            })
        return results

    def same_college_connections(self, player_id: str) -> list[dict]:
        """
        Find other NFL players who attended the same college.
        Highlights the college-to-NFL pipeline and whether
        college teammates end up in the same coaching systems.
        """
        self._require_built()
        p = self.players.get(player_id)
        if not p or not p.college:
            return []
        return [
            {
                "player_id": pid,
                "name": other.name,
                "position": other.primary_position,
                "college": other.college,
                "teams": list(set(other.teams)),
                "career_span": other.career_span,
            }
            for pid, other in self.players.items()
            if pid != player_id and other.college.lower() == p.college.lower()
        ]

    # ── Graph stats ───────────────────────────────────────────────────────────

    def summary_stats(self) -> dict:
        """Return high-level statistics about the graph."""
        self._require_built()
        player_nodes = [n for n, d in self.G.nodes(data=True)
                        if d.get("type") == "player"]
        coach_nodes  = [n for n, d in self.G.nodes(data=True)
                        if d.get("type") == "coach"]

        # Largest connected component (players only projected graph)
        try:
            components = list(nx.connected_components(self.G))
            largest_cc = max(components, key=len)
            lcc_players = len([n for n in largest_cc
                               if not n.startswith(self.COACH_PREFIX)])
        except Exception:
            lcc_players = 0

        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "player_count": len(player_nodes),
            "coach_count":  len(coach_nodes),
            "largest_component_players": lcc_players,
            "avg_degree": round(
                sum(d for _, d in self.G.degree()) / max(1, self.G.number_of_nodes()), 2
            ),
            "density": round(nx.density(self.G), 4),
        }

    # ── NetworkX projection for visualisation ────────────────────────────────

    def player_projection(self) -> nx.Graph:
        """
        Return a player-only graph where two players are connected
        if they share at least one coach. Edge weight = shared coaches.
        Useful for visualization.
        """
        self._require_built()
        proj = nx.Graph()
        for pid, p in self.players.items():
            proj.add_node(pid,
                          label=p.name,
                          position=p.primary_position,
                          teams=list(set(p.teams)))

        # For every coach node, connect all its player neighbors
        for node, data in self.G.nodes(data=True):
            if data.get("type") == "coach":
                neighbors = [n for n in self.G.neighbors(node)
                             if not n.startswith(self.COACH_PREFIX)]
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        u, v = neighbors[i], neighbors[j]
                        if proj.has_edge(u, v):
                            proj[u][v]["weight"] += 1
                        else:
                            proj.add_edge(u, v, weight=1)
        return proj
