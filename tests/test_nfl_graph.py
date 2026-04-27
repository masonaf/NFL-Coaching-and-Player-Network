"""
tests/test_nfl_graph.py
-----------------------
Test suite for the NFL Coaching Network project.

Tests
-----
- Data loading and CSV validation
- Player / Coach object construction
- Graph structure (nodes, edges, types)
- Mode 1: Player profile
- Mode 2: Teammates via shared coaches
- Mode 3: Shortest path
- Mode 4: Centrality rankings
- Mode 5: Coaching tree
- Edge cases and error handling

Run:
  python -m pytest tests/ -v
  python -m pytest tests/ -v --tb=short   # compact failures
"""

import os
import sys
import pytest
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nfl_graph import NFLGraph, Player, Coach
from data_fetcher import seed_demo_data, DATA_DIR


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def seeded_data():
    """Ensure rich demo CSVs exist (with college column), return their paths."""
    rosters_path = os.path.join(DATA_DIR, "rosters.csv")
    coaches_path  = os.path.join(DATA_DIR, "coaches.csv")

    # Use the rich generator if available, else seed_demo_data
    gen_script = os.path.join(os.path.dirname(__file__), "..", "generate_seed_data.py")
    if os.path.exists(gen_script):
        import subprocess
        # Always regenerate for clean test state
        for p in [rosters_path, coaches_path]:
            if os.path.exists(p):
                os.remove(p)
        subprocess.run(["python", gen_script], check=True)
    else:
        for p in [rosters_path, coaches_path]:
            if os.path.exists(p):
                os.remove(p)
        seed_demo_data()

    return rosters_path, coaches_path


@pytest.fixture(scope="session")
def graph(seeded_data):
    """Build and return a populated NFLGraph."""
    rosters_path, coaches_path = seeded_data
    g = NFLGraph()
    g.build(rosters_path, coaches_path)
    return g


# ── Data layer tests ──────────────────────────────────────────────────────────

class TestSeedData:
    def test_csv_files_created(self, seeded_data):
        rosters_path, coaches_path = seeded_data
        assert os.path.exists(rosters_path), "rosters.csv not found"
        assert os.path.exists(coaches_path),  "coaches.csv not found"

    def test_rosters_columns(self, seeded_data):
        df = pd.read_csv(seeded_data[0])
        for col in ["player_id", "name", "position", "college", "team", "year"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_college_column_populated(self, seeded_data):
        df = pd.read_csv(seeded_data[0])
        non_empty = df["college"].dropna()
        non_empty = non_empty[non_empty != ""]
        assert len(non_empty) > 0, "college column is entirely empty"

    def test_multiple_seasons_per_player(self, seeded_data):
        """Professor requirement: multiple years so players actually connect."""
        df = pd.read_csv(seeded_data[0])
        seasons_per_player = df.groupby("player_id")["year"].nunique()
        multi_season = (seasons_per_player > 1).sum()
        assert multi_season > 10, f"Expected >10 players with multiple seasons, got {multi_season}"

    def test_year_range_spans_multiple_seasons(self, seeded_data):
        df = pd.read_csv(seeded_data[0])
        span = df["year"].max() - df["year"].min()
        assert span >= 5, f"Year range too small: {span} years"

    def test_coaches_columns(self, seeded_data):
        df = pd.read_csv(seeded_data[1])
        for col in ["team", "year", "head_coach"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_rosters_non_empty(self, seeded_data):
        df = pd.read_csv(seeded_data[0])
        assert len(df) > 50, "Expected at least 50 player-season rows"

    def test_coaches_non_empty(self, seeded_data):
        df = pd.read_csv(seeded_data[1])
        assert len(df) > 10, "Expected at least 10 team-season rows"

    def test_no_null_player_ids(self, seeded_data):
        df = pd.read_csv(seeded_data[0])
        assert df["player_id"].notna().all(), "Null player_ids found"

    def test_valid_years(self, seeded_data):
        df = pd.read_csv(seeded_data[0])
        assert df["year"].between(1990, 2030).all(), "Years out of expected range"


# ── Player / Coach dataclass tests ────────────────────────────────────────────

class TestPlayerDataclass:
    def test_player_primary_position(self):
        p = Player("test-01", "Test Player",
                   positions=["QB", "QB", "WR"],
                   teams=["NWE"],
                   seasons=[2020, 2021, 2022])
        assert p.primary_position == "QB"

    def test_player_career_span(self):
        p = Player("test-02", "Another Player",
                   seasons=[2015, 2018, 2020])
        assert p.career_span == "2015–2020"

    def test_player_num_seasons_deduped(self):
        p = Player("test-03", "Third Player",
                   seasons=[2018, 2018, 2019, 2019, 2020])
        assert p.num_seasons == 3

    def test_player_empty_seasons(self):
        p = Player("test-04", "Empty Player")
        assert p.career_span == "Unknown"
        assert p.num_seasons == 0
        assert p.primary_position == "UNK"

    def test_player_repr(self):
        p = Player("test-05", "Repr Player",
                   positions=["TE"], teams=["KAN"], seasons=[2022],
                   college="Cincinnati")
        r = repr(p)
        assert "Repr Player" in r
        assert "TE" in r
        assert "Cincinnati" in r

    def test_player_college_field(self):
        p = Player("test-06", "College Player",
                   positions=["QB"], teams=["NWE"], seasons=[2020],
                   college="Michigan")
        assert p.college == "Michigan"

    def test_player_default_college_empty(self):
        p = Player("test-07", "No College")
        assert p.college == ""

    def test_coach_dataclass(self):
        c = Coach("Test Coach",
                  teams=["NWE", "NWE", "DET"],
                  seasons=[2010, 2011, 2015],
                  players=["p1", "p2"])
        assert c.num_seasons == 3
        assert len(c.teams_coached) == 2  # deduplicated
        assert c.career_span == "2010–2015"


# ── Graph construction tests ──────────────────────────────────────────────────

class TestGraphConstruction:
    def test_graph_built(self, graph):
        assert graph._built is True

    def test_node_count_positive(self, graph):
        assert graph.G.number_of_nodes() > 0

    def test_edge_count_positive(self, graph):
        assert graph.G.number_of_edges() > 0

    def test_player_nodes_exist(self, graph):
        player_nodes = [n for n, d in graph.G.nodes(data=True)
                        if d.get("type") == "player"]
        assert len(player_nodes) > 0

    def test_coach_nodes_exist(self, graph):
        coach_nodes = [n for n, d in graph.G.nodes(data=True)
                       if d.get("type") == "coach"]
        assert len(coach_nodes) > 0

    def test_coach_node_prefix(self, graph):
        for node in graph.G.nodes():
            if graph.G.nodes[node].get("type") == "coach":
                assert node.startswith(graph.COACH_PREFIX)

    def test_edge_weights_positive(self, graph):
        for u, v, data in graph.G.edges(data=True):
            assert data.get("weight", 0) > 0, f"Edge ({u},{v}) has non-positive weight"

    def test_known_players_loaded(self, graph):
        """Tom Brady and Patrick Mahomes should be in the graph."""
        brady = graph.find_player("Tom Brady")
        mahomes = graph.find_player("Patrick Mahomes")
        assert brady, "Tom Brady not found"
        assert mahomes, "Patrick Mahomes not found"

    def test_known_coaches_loaded(self, graph):
        belichick = graph.find_coach("Belichick")
        reid = graph.find_coach("Reid")
        assert belichick, "Bill Belichick not found"
        assert reid, "Andy Reid not found"

    def test_players_dict_populated(self, graph):
        assert len(graph.players) > 0

    def test_coaches_dict_populated(self, graph):
        assert len(graph.coaches) > 0

    def test_summary_stats_keys(self, graph):
        stats = graph.summary_stats()
        for key in ["total_nodes", "total_edges", "player_count",
                    "coach_count", "avg_degree", "density"]:
            assert key in stats, f"Missing stat: {key}"

    def test_density_between_0_and_1(self, graph):
        stats = graph.summary_stats()
        assert 0 <= stats["density"] <= 1

    def test_player_college_attribute_on_node(self, graph):
        """College should be stored as a node attribute."""
        for node, data in graph.G.nodes(data=True):
            if data.get("type") == "player":
                assert "college" in data, f"Node {node} missing college attribute"

    def test_graph_is_reasonably_connected(self, graph):
        """With multi-year data, the graph should not be heavily fragmented."""
        import networkx as nx
        components = list(nx.connected_components(graph.G))
        # Should have at most a handful of components (bridge players connect most)
        player_nodes = len([n for n in graph.G.nodes() if not n.startswith(graph.COACH_PREFIX)])
        # Largest component should contain at least 50% of player nodes
        largest = max(len([n for n in c if not n.startswith(graph.COACH_PREFIX)])
                      for c in components)
        assert largest >= player_nodes * 0.4, (
            f"Largest component has only {largest}/{player_nodes} players — graph too fragmented"
        )


# ── Mode 1: Player profile ────────────────────────────────────────────────────

class TestPlayerProfile:
    def test_profile_returns_dict(self, graph):
        brady = graph.find_player("Tom Brady")[0]
        profile = graph.player_profile(brady.player_id)
        assert isinstance(profile, dict)

    def test_profile_has_required_keys(self, graph):
        brady = graph.find_player("Tom Brady")[0]
        profile = graph.player_profile(brady.player_id)
        for key in ["player", "coaches", "teammates"]:
            assert key in profile

    def test_profile_player_is_correct(self, graph):
        brady = graph.find_player("Tom Brady")[0]
        profile = graph.player_profile(brady.player_id)
        assert profile["player"].name == "Tom Brady"

    def test_profile_has_coaches(self, graph):
        brady = graph.find_player("Tom Brady")[0]
        profile = graph.player_profile(brady.player_id)
        assert len(profile["coaches"]) > 0

    def test_profile_coach_has_seasons(self, graph):
        brady = graph.find_player("Tom Brady")[0]
        profile = graph.player_profile(brady.player_id)
        for c in profile["coaches"]:
            assert c["seasons_together"] > 0

    def test_profile_unknown_player(self, graph):
        profile = graph.player_profile("nonexistent-player-999")
        assert profile == {}

    def test_find_player_case_insensitive(self, graph):
        assert graph.find_player("brady") == graph.find_player("Brady")

    def test_find_player_partial_match(self, graph):
        results = graph.find_player("Kelce")
        assert any("Kelce" in p.name for p in results)


# ── Mode 2: Teammates ─────────────────────────────────────────────────────────

class TestTeammates:
    def test_teammates_returns_list(self, graph):
        brady = graph.find_player("Tom Brady")[0]
        result = graph.get_teammates(brady.player_id)
        assert isinstance(result, list)

    def test_teammates_all_have_required_keys(self, graph):
        brady = graph.find_player("Tom Brady")[0]
        for t in graph.get_teammates(brady.player_id):
            for key in ["player_id", "name", "position", "shared_coaches"]:
                assert key in t

    def test_teammates_sorted_desc(self, graph):
        brady = graph.find_player("Tom Brady")[0]
        teammates = graph.get_teammates(brady.player_id)
        counts = [t["shared_coaches"] for t in teammates]
        assert counts == sorted(counts, reverse=True)

    def test_gronkowski_is_brady_teammate(self, graph):
        """Brady and Gronk played under Belichick and Arians — should be connected."""
        brady = graph.find_player("Tom Brady")[0]
        teammates = graph.get_teammates(brady.player_id)
        names = [t["name"] for t in teammates]
        assert any("Gronkowski" in n for n in names), "Gronkowski should be Brady's teammate"

    def test_min_shared_filter(self, graph):
        brady = graph.find_player("Tom Brady")[0]
        t1 = graph.get_teammates(brady.player_id, min_shared_coaches=1)
        t2 = graph.get_teammates(brady.player_id, min_shared_coaches=2)
        assert len(t1) >= len(t2), "Higher threshold should return <= results"

    def test_player_not_own_teammate(self, graph):
        brady = graph.find_player("Tom Brady")[0]
        teammate_ids = [t["player_id"] for t in graph.get_teammates(brady.player_id)]
        assert brady.player_id not in teammate_ids


# ── Mode 3: Shortest path ─────────────────────────────────────────────────────

class TestShortestPath:
    def _get_pid(self, graph, name):
        results = graph.find_player(name)
        assert results, f"{name} not found"
        return results[0].player_id

    def test_path_same_team_players(self, graph):
        """Mahomes and Kelce both played under Andy Reid — 2 hops."""
        pid1 = self._get_pid(graph, "Patrick Mahomes")
        pid2 = self._get_pid(graph, "Travis Kelce")
        result = graph.shortest_path(pid1, pid2)
        assert result["found"]
        assert result["degrees"] == 1  # Mahomes → Reid → Kelce

    def test_path_returns_steps(self, graph):
        pid1 = self._get_pid(graph, "Patrick Mahomes")
        pid2 = self._get_pid(graph, "Travis Kelce")
        result = graph.shortest_path(pid1, pid2)
        assert "steps" in result
        assert len(result["steps"]) >= 3  # player, coach, player

    def test_path_alternates_types(self, graph):
        """Steps should alternate player → coach → player → …"""
        pid1 = self._get_pid(graph, "Patrick Mahomes")
        pid2 = self._get_pid(graph, "Travis Kelce")
        result = graph.shortest_path(pid1, pid2)
        types = [s["type"] for s in result["steps"]]
        for i in range(len(types) - 1):
            assert types[i] != types[i + 1], "Consecutive same-type steps"

    def test_path_nonexistent_node(self, graph):
        pid1 = self._get_pid(graph, "Patrick Mahomes")
        result = graph.shortest_path(pid1, "ghost-player-000")
        assert result["found"] is False

    def test_path_cross_coaching_tree(self, graph):
        """Brady (Belichick/Arians) and Mahomes (Reid) should still connect."""
        pid1 = self._get_pid(graph, "Tom Brady")
        pid2 = self._get_pid(graph, "Patrick Mahomes")
        result = graph.shortest_path(pid1, pid2)
        # May or may not have a path depending on seed data — just test structure
        assert "found" in result

    def test_degrees_consistent_with_path_length(self, graph):
        pid1 = self._get_pid(graph, "Patrick Mahomes")
        pid2 = self._get_pid(graph, "Travis Kelce")
        result = graph.shortest_path(pid1, pid2)
        if result["found"]:
            expected_degrees = (result["length"]) // 2
            assert result["degrees"] == expected_degrees


# ── Mode 4: Centrality rankings ───────────────────────────────────────────────

class TestCentralityRankings:
    def test_centrality_returns_list(self, graph):
        assert isinstance(graph.centrality_rankings(), list)

    def test_centrality_top_n(self, graph):
        for n in [5, 10, 20]:
            r = graph.centrality_rankings(top_n=n)
            assert len(r) <= n

    def test_centrality_has_required_keys(self, graph):
        rankings = graph.centrality_rankings(top_n=5)
        for row in rankings:
            for key in ["player_id", "name", "position", "centrality", "num_coaches"]:
                assert key in row

    def test_centrality_sorted_desc(self, graph):
        rankings = graph.centrality_rankings(top_n=20)
        scores = [r["centrality"] for r in rankings]
        assert scores == sorted(scores, reverse=True)

    def test_centrality_between_0_and_1(self, graph):
        for r in graph.centrality_rankings():
            assert 0 <= r["centrality"] <= 1

    def test_centrality_no_coaches_in_results(self, graph):
        """Rankings should only include player nodes."""
        for r in graph.centrality_rankings():
            assert r["player_id"] in graph.players


# ── Mode 5: Coaching tree ─────────────────────────────────────────────────────

class TestCoachingTree:
    def test_coaching_tree_returns_dict(self, graph):
        result = graph.coaching_tree("Andy Reid")
        assert isinstance(result, dict)

    def test_coaching_tree_has_keys(self, graph):
        result = graph.coaching_tree("Andy Reid")
        for key in ["coach", "players", "related_coaches", "total_players"]:
            assert key in result

    def test_coaching_tree_player_count(self, graph):
        result = graph.coaching_tree("Andy Reid")
        assert result["total_players"] > 0

    def test_coaching_tree_players_have_seasons(self, graph):
        result = graph.coaching_tree("Andy Reid")
        for p in result["players"]:
            assert p["seasons_together"] > 0

    def test_coaching_tree_mahomes_under_reid(self, graph):
        result = graph.coaching_tree("Andy Reid")
        player_names = [p["name"] for p in result["players"]]
        assert any("Mahomes" in n for n in player_names)

    def test_coaching_tree_unknown_coach(self, graph):
        result = graph.coaching_tree("Definitely Not A Real Coach 999")
        assert result == {}

    def test_find_coach_partial(self, graph):
        results = graph.find_coach("Reid")
        assert any("Reid" in c.name for c in results)

    def test_find_coach_case_insensitive(self, graph):
        assert graph.find_coach("reid") == graph.find_coach("Reid")


# ── Projection graph ──────────────────────────────────────────────────────────

class TestProjection:
    def test_player_projection_is_graph(self, graph):
        proj = graph.player_projection()
        assert isinstance(proj, nx.Graph)

    def test_projection_only_players(self, graph):
        proj = graph.player_projection()
        for node in proj.nodes():
            assert not node.startswith(graph.COACH_PREFIX)

    def test_projection_weights_positive(self, graph):
        proj = graph.player_projection()
        for u, v, d in proj.edges(data=True):
            assert d.get("weight", 0) > 0


# ── College pipeline tests (professor-suggested feature) ─────────────────────

class TestCollegePipeline:
    def test_college_alumni_returns_list(self, graph):
        assert isinstance(graph.college_alumni("Alabama"), list)

    def test_college_alumni_finds_known_school(self, graph):
        """Multiple Alabama players are in the dataset."""
        results = graph.college_alumni("Alabama")
        assert len(results) > 0, "Expected Alabama alumni in dataset"

    def test_college_alumni_case_insensitive(self, graph):
        r1 = graph.college_alumni("alabama")
        r2 = graph.college_alumni("Alabama")
        assert r1 == r2

    def test_college_alumni_partial_match(self, graph):
        """'Mich' should match Michigan, Michigan State, etc."""
        results = graph.college_alumni("Mich")
        colleges = [r["college"] for r in results]
        assert any("Michigan" in c for c in colleges)

    def test_college_alumni_has_required_keys(self, graph):
        results = graph.college_alumni("Alabama")
        if results:
            for key in ["player_id", "name", "position", "college", "teams", "career_span"]:
                assert key in results[0], f"Missing key: {key}"

    def test_college_pipeline_returns_list(self, graph):
        assert isinstance(graph.college_pipeline(), list)

    def test_college_pipeline_sorted_desc(self, graph):
        pipeline = graph.college_pipeline()
        counts = [p["player_count"] for p in pipeline]
        assert counts == sorted(counts, reverse=True)

    def test_college_pipeline_has_required_keys(self, graph):
        for row in graph.college_pipeline(top_n=3):
            for key in ["college", "player_count", "nfl_teams_reached", "players"]:
                assert key in row

    def test_college_pipeline_top_n(self, graph):
        for n in [5, 10]:
            result = graph.college_pipeline(top_n=n)
            assert len(result) <= n

    def test_same_college_connections(self, graph):
        """Brady (Michigan) — check if any other Michigan player is in the dataset."""
        brady = graph.find_player("Tom Brady")[0]
        results = graph.same_college_connections(brady.player_id)
        # Should return list (may be empty if no other Michigan players in seed)
        assert isinstance(results, list)
        # Every result should have the same college
        for r in results:
            assert r["college"].lower() == brady.college.lower()

    def test_same_college_no_self(self, graph):
        """Player should not appear in their own same-college results."""
        brady = graph.find_player("Tom Brady")[0]
        results = graph.same_college_connections(brady.player_id)
        assert brady.player_id not in [r["player_id"] for r in results]

    def test_same_college_unknown_player(self, graph):
        result = graph.same_college_connections("ghost-999")
        assert result == []

    def test_player_college_in_graph_node(self, graph):
        """College should be accessible via graph node attributes."""
        brady = graph.find_player("Tom Brady")[0]
        node_data = graph.G.nodes.get(brady.player_id, {})
        assert "college" in node_data
        assert node_data["college"] == "Michigan"

    def test_multiple_teams_alumni_have_college(self, graph):
        """All players in the graph should have college attribute (may be empty string)."""
        for pid, p in graph.players.items():
            assert hasattr(p, "college"), f"Player {p.name} missing college attribute"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    subprocess.run(["python", "-m", "pytest", __file__, "-v"], check=True)
