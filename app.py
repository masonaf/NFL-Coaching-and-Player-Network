"""
app.py  —  NFL Coaching Network Explorer
==========================================
Streamlit application for SI 507 Final Project.

Run:
  streamlit run app.py

Interaction Modes
-----------------
  1. Search & Player Profile
  2. Teammates via Shared Coaches
  3. Shortest Path (Six Degrees of NFL)
  4. Centrality Rankings
  5. Coaching Tree Explorer  (bonus)
"""

import os
import sys
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

# Local modules
sys.path.insert(0, os.path.dirname(__file__))
from nfl_graph import NFLGraph, Player, Coach
from data_fetcher import seed_demo_data, DATA_DIR


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NFL Coaching Network",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }
  .main-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    letter-spacing: 3px;
    color: #E31837;
    margin-bottom: 0;
  }
  .subtitle {
    color: #888;
    font-size: 0.95rem;
    margin-top: 0;
    margin-bottom: 1.5rem;
  }
  .stat-card {
    background: #1a1a2e;
    border-left: 4px solid #E31837;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 10px;
  }
  .stat-card h4 { margin: 0 0 4px 0; color: #E31837; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
  .stat-card p  { margin: 0; font-size: 1.4rem; font-weight: 600; color: #fff; }
  .path-step-player { background:#E31837; color:#fff; padding:4px 12px; border-radius:20px; display:inline-block; margin:4px; }
  .path-step-coach  { background:#1a1a2e; color:#ddd; border:1px solid #E31837; padding:4px 12px; border-radius:20px; display:inline-block; margin:4px; }
  .arrow { color: #E31837; font-size: 1.2rem; margin: 0 4px; }
  div[data-testid="metric-container"] { background:#1a1a2e; border-radius:8px; padding:12px; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Building coaching network graph …")
def load_graph() -> NFLGraph:
    rosters_path = os.path.join(DATA_DIR, "rosters.csv")
    coaches_path  = os.path.join(DATA_DIR, "coaches.csv")

    # Auto-seed demo data if nothing exists
    if not os.path.exists(rosters_path) or not os.path.exists(coaches_path):
        seed_demo_data()

    g = NFLGraph()
    g.build(rosters_path, coaches_path)
    return g


# ── Visualisation helpers ─────────────────────────────────────────────────────

def draw_player_network(g: NFLGraph, center_pid: str, depth: int = 1) -> go.Figure:
    """Draw the ego network around a player using Plotly."""
    ego = nx.ego_graph(g.G, center_pid, radius=depth)
    pos = nx.spring_layout(ego, seed=42, k=1.2)

    edge_x, edge_y = [], []
    for u, v in ego.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=1, color="#555"),
                            hoverinfo="none", showlegend=False)

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in ego.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        data = ego.nodes[node]
        label = data.get("label", node)
        ntype = data.get("type", "player")
        node_text.append(label)
        if node == center_pid:
            node_color.append("#E31837"); node_size.append(22)
        elif ntype == "coach":
            node_color.append("#FFD700"); node_size.append(16)
        else:
            node_color.append("#4a90d9"); node_size.append(12)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_text, textposition="top center",
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color="#333")),
        hoverinfo="text", showlegend=False
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#0d0d1a",
        font_color="#eee",
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def draw_path(steps: list[dict]) -> str:
    """Render HTML for shortest-path visualization."""
    parts = []
    for i, step in enumerate(steps):
        cls = "path-step-player" if step["type"] == "player" else "path-step-coach"
        icon = "🏈 " if step["type"] == "player" else "🎯 "
        parts.append(f'<span class="{cls}">{icon}{step["label"]}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="arrow">→</span>')
    return " ".join(parts)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="main-title">🏈 NFL<br>COACHING<br>NETWORK</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">SI 507 Final Project</p>', unsafe_allow_html=True)
    st.divider()

    mode = st.radio("Select Mode", [
        "📋 Player Profile",
        "🤝 Teammates via Coaches",
        "🔗 Shortest Path",
        "📊 Centrality Rankings",
        "🌳 Coaching Tree",
        "🎓 College Pipeline",
    ], index=0)

    st.divider()
    st.caption("**About this graph**")
    st.caption("Nodes = players & coaches  \nEdges = seasons played together  \nWeight = years under same coach  \nAttr = college, position, team history")


# ── Load graph ────────────────────────────────────────────────────────────────

g = load_graph()
stats = g.summary_stats()

# Top stats bar
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Players",   stats["player_count"])
c2.metric("Coaches",   stats["coach_count"])
c3.metric("Edges",     stats["total_edges"])
c4.metric("Avg Degree", stats["avg_degree"])
c5.metric("Density",   stats["density"])

st.divider()

# ── Player name lookup helper ─────────────────────────────────────────────────

def player_selectbox(label: str, key: str) -> str | None:
    """Search box + selectbox that returns a player_id."""
    query = st.text_input(label, key=f"{key}_q", placeholder="e.g. Brady, Mahomes …")
    if not query:
        return None
    matches = g.find_player(query)
    if not matches:
        st.warning("No players found.")
        return None
    options = {f"{p.name} ({p.primary_position}, {p.career_span})": p.player_id
               for p in matches}
    chosen_label = st.selectbox("Select player", list(options.keys()), key=f"{key}_sel")
    return options[chosen_label]


# ═══════════════════════════════════════════════════════════════════
# MODE 1: Player Profile
# ═══════════════════════════════════════════════════════════════════

if mode == "📋 Player Profile":
    st.subheader("Player Profile")
    st.caption("Search for any player to see their career, coaching history, and network position.")

    pid = player_selectbox("Search player", "profile")

    if pid:
        profile = g.player_profile(pid)
        p: Player = profile["player"]

        col_info, col_graph = st.columns([1, 2])

        with col_info:
            st.markdown(f"### {p.name}")
            st.markdown(f"**Position:** {p.primary_position}")
            st.markdown(f"**College:** {p.college or '—'}")
            st.markdown(f"**Career:** {p.career_span} ({p.num_seasons} seasons)")
            st.markdown(f"**Teams:** {', '.join(sorted(set(p.teams)))}")

            st.markdown("#### Coaches")
            for c_info in profile["coaches"]:
                st.markdown(f"- **{c_info['coach']}** ({c_info['seasons_together']} season{'s' if c_info['seasons_together'] > 1 else ''})")

            st.markdown("#### Top Connections")
            for tm in profile["teammates"][:6]:
                st.markdown(f"- {tm['player']} *(shared {tm['shared_coaches']} coach{'es' if tm['shared_coaches'] > 1 else ''})*")

        with col_graph:
            st.caption("🔴 Selected player  🟡 Coaches  🔵 Connected players")
            fig = draw_player_network(g, pid, depth=2)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# MODE 2: Teammates via Coaches
# ═══════════════════════════════════════════════════════════════════

elif mode == "🤝 Teammates via Coaches":
    st.subheader("Teammates via Shared Coaches")
    st.caption("Find players connected through the coaching network, revealing system families across teams.")

    pid = player_selectbox("Search player", "teammates")
    min_shared = st.slider("Minimum shared coaches", 1, 5, 1)

    if pid:
        p = g.players.get(pid)
        st.markdown(f"**Showing connections for: {p.name if p else pid}**")

        teammates = g.get_teammates(pid, min_shared_coaches=min_shared)

        if not teammates:
            st.info("No teammates found with those filters.")
        else:
            df = pd.DataFrame(teammates)[["name", "position", "shared_coaches", "teams"]]
            df["teams"] = df["teams"].apply(lambda t: ", ".join(sorted(t)))
            df.columns = ["Player", "Position", "Shared Coaches", "Teams"]
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Bar chart
            top10 = teammates[:10]
            fig = px.bar(
                x=[t["name"] for t in top10],
                y=[t["shared_coaches"] for t in top10],
                labels={"x": "Player", "y": "Shared Coaches"},
                color=[t["shared_coaches"] for t in top10],
                color_continuous_scale=["#1a1a2e", "#E31837"],
            )
            fig.update_layout(
                paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
                font_color="#eee", coloraxis_showscale=False,
                xaxis_tickangle=-35, height=320
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# MODE 3: Shortest Path
# ═══════════════════════════════════════════════════════════════════

elif mode == "🔗 Shortest Path":
    st.subheader("Six Degrees of NFL — Shortest Path")
    st.caption("Find the shortest connection between any two players through the coaching network.")

    col_a, col_b = st.columns(2)
    with col_a:
        pid1 = player_selectbox("Player 1", "path_a")
    with col_b:
        pid2 = player_selectbox("Player 2", "path_b")

    if pid1 and pid2:
        if pid1 == pid2:
            st.warning("Please select two different players.")
        else:
            result = g.shortest_path(pid1, pid2)

            if not result["found"]:
                st.error(f"No connection found: {result['reason']}")
            else:
                p1 = g.players.get(pid1)
                p2 = g.players.get(pid2)
                st.success(
                    f"**{p1.name if p1 else pid1}** → **{p2.name if p2 else pid2}**  |  "
                    f"**{result['degrees']} degree{'s' if result['degrees'] != 1 else ''} of separation**"
                )
                st.markdown("#### Connection Path")
                st.markdown(draw_path(result["steps"]), unsafe_allow_html=True)

                st.markdown("#### Step-by-step explanation")
                steps = result["steps"]
                for i in range(0, len(steps) - 1, 2):
                    player_step = steps[i]
                    if i + 1 < len(steps):
                        coach_step = steps[i + 1]
                        if i + 2 < len(steps):
                            next_player = steps[i + 2]
                            st.markdown(
                                f"- **{player_step['label']}** played under coach **{coach_step['label']}**, "
                                f"who also coached **{next_player['label']}**"
                            )


# ═══════════════════════════════════════════════════════════════════
# MODE 4: Centrality Rankings
# ═══════════════════════════════════════════════════════════════════

elif mode == "📊 Centrality Rankings":
    st.subheader("Most Connected Players — Centrality Rankings")
    st.caption(
        "Degree centrality measures how many unique coaches a player connected through. "
        "High centrality = journeyman linchpins who bridge multiple coaching systems."
    )

    top_n = st.slider("Show top N players", 5, 50, 20)
    rankings = g.centrality_rankings(top_n=top_n)

    df = pd.DataFrame(rankings)
    df["teams"] = df["teams"].apply(lambda t: ", ".join(sorted(t)))
    df.index = range(1, len(df) + 1)
    display_df = df[["name", "position", "centrality", "num_coaches", "career_span", "teams"]]
    display_df.columns = ["Player", "Pos", "Centrality", "# Coaches", "Career", "Teams"]

    st.dataframe(display_df, use_container_width=True)

    # Lollipop chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["centrality"],
        y=df["name"],
        mode="markers",
        marker=dict(size=12, color="#E31837"),
        name=""
    ))
    for _, row in df.iterrows():
        fig.add_shape(type="line",
                      x0=0, x1=row["centrality"],
                      y0=row["name"], y1=row["name"],
                      line=dict(color="#E31837", width=1.5))
    fig.update_layout(
        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
        font_color="#eee", height=max(350, top_n * 22),
        xaxis_title="Degree Centrality",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# MODE 5: Coaching Tree (Bonus / A+ Feature)
# ═══════════════════════════════════════════════════════════════════

elif mode == "🌳 Coaching Tree":
    st.subheader("Coaching Tree Explorer")
    st.caption(
        "Explore a coach's full network: who they coached, how long, and which other coaches "
        "share players with them — revealing NFL coaching family trees."
    )

    query = st.text_input("Search coach", placeholder="e.g. Belichick, Reid, McVay …")
    if query:
        matches = g.find_coach(query)
        if not matches:
            st.warning("No coaches found.")
        else:
            options = {f"{c.name} ({c.career_span})": c.name for c in matches}
            chosen = st.selectbox("Select coach", list(options.keys()))
            coach_name = options[chosen]

            tree = g.coaching_tree(coach_name)
            if not tree:
                st.error("Coach not found in graph.")
            else:
                coach: Coach = tree["coach"]

                col1, col2, col3 = st.columns(3)
                col1.metric("Players Coached", tree["total_players"])
                col2.metric("Teams",           len(coach.teams_coached))
                col3.metric("Seasons (HC)",    coach.num_seasons)

                col_players, col_related = st.columns([1, 1])

                with col_players:
                    st.markdown("#### Players Coached")
                    df_p = pd.DataFrame(tree["players"])[["name", "position", "seasons_together"]]
                    df_p.columns = ["Player", "Position", "Seasons Together"]
                    st.dataframe(df_p, use_container_width=True, hide_index=True)

                with col_related:
                    st.markdown("#### Related Coaches (share players)")
                    st.caption("Coaches who coached the same players — the coaching family tree.")
                    df_c = pd.DataFrame(tree["related_coaches"])
                    df_c.columns = ["Coach", "Shared Players"]
                    st.dataframe(df_c, use_container_width=True, hide_index=True)

                # Network viz — ego graph around the coach node
                coach_node = g.COACH_PREFIX + coach_name
                if coach_node in g.G:
                    ego = nx.ego_graph(g.G, coach_node, radius=2)
                    pos = nx.spring_layout(ego, seed=7, k=1.5)

                    ex, ey, nx_, ny_, labels_, colors_, sizes_ = [], [], [], [], [], [], []
                    for u, v in ego.edges():
                        x0, y0 = pos[u]; x1, y1 = pos[v]
                        ex += [x0, x1, None]; ey += [y0, y1, None]
                    for node in ego.nodes():
                        x, y = pos[node]
                        nx_.append(x); ny_.append(y)
                        data = ego.nodes[node]
                        lbl = data.get("label", node)
                        labels_.append(lbl)
                        if node == coach_node:
                            colors_.append("#FFD700"); sizes_.append(24)
                        elif data.get("type") == "coach":
                            colors_.append("#FFA500"); sizes_.append(16)
                        else:
                            colors_.append("#4a90d9"); sizes_.append(10)

                    fig = go.Figure(data=[
                        go.Scatter(x=ex, y=ey, mode="lines",
                                   line=dict(width=1, color="#444"), hoverinfo="none", showlegend=False),
                        go.Scatter(x=nx_, y=ny_, mode="markers+text", text=labels_,
                                   textposition="top center",
                                   marker=dict(size=sizes_, color=colors_,
                                               line=dict(width=1, color="#222")),
                                   hoverinfo="text", showlegend=False)
                    ])
                    fig.update_layout(
                        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
                        font_color="#eee", height=450,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.caption("🟡 Selected coach  🟠 Related coaches  🔵 Players")
                    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# MODE 6: College Pipeline (professor-suggested feature)
# ═══════════════════════════════════════════════════════════════════

elif mode == "🎓 College Pipeline":
    st.subheader("College → NFL Pipeline")
    st.caption(
        "Which colleges produce the most NFL players in this dataset? "
        "How are college teammates distributed across coaching systems? "
        "Search a college to see all its NFL alumni."
    )

    tab1, tab2 = st.tabs(["🏆 Top Colleges by NFL Players", "🔍 College Alumni Search"])

    with tab1:
        top_n_col = st.slider("Show top N colleges", 5, 30, 15)
        pipeline = g.college_pipeline(top_n=top_n_col)

        if not pipeline:
            st.info("No college data available. Run the scraper or use the rich seed dataset.")
        else:
            df_pipe = pd.DataFrame(pipeline)[["college", "player_count", "nfl_teams_reached"]]
            df_pipe.index = range(1, len(df_pipe) + 1)
            df_pipe.columns = ["College", "NFL Players", "Teams Reached"]

            col_table, col_chart = st.columns([1, 1])

            with col_table:
                st.dataframe(df_pipe, use_container_width=True)

            with col_chart:
                fig = px.bar(
                    x=[p["player_count"] for p in pipeline],
                    y=[p["college"] for p in pipeline],
                    orientation="h",
                    color=[p["nfl_teams_reached"] for p in pipeline],
                    color_continuous_scale=["#1a1a2e", "#E31837"],
                    labels={"x": "NFL Players", "y": "College",
                            "color": "Teams Reached"},
                )
                fig.update_layout(
                    paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
                    font_color="#eee", height=420,
                    yaxis=dict(autorange="reversed"),
                    coloraxis_colorbar=dict(title="Teams"),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Key insight callout
            if pipeline:
                top = pipeline[0]
                st.info(
                    f"**{top['college']}** leads with **{top['player_count']} NFL players** "
                    f"across **{top['nfl_teams_reached']} teams** in this dataset. "
                    f"Notable alumni: {', '.join(top['players'][:4])}{'…' if len(top['players']) > 4 else ''}."
                )

    with tab2:
        col_query = st.text_input("Search college", placeholder="e.g. Alabama, Michigan, LSU …")

        if col_query:
            alumni = g.college_alumni(col_query)
            if not alumni:
                st.warning(f"No players found from colleges matching '{col_query}'.")
            else:
                # Show which college(s) matched
                matched_colleges = sorted(set(a["college"] for a in alumni))
                st.success(f"Found **{len(alumni)} players** from: {', '.join(matched_colleges)}")

                df_al = pd.DataFrame(alumni)
                df_al["teams"] = df_al["teams"].apply(lambda t: ", ".join(sorted(t)))
                df_al["coaches"] = df_al["coaches"].apply(
                    lambda c: ", ".join(sorted(set(c)))[:60] + ("…" if len(set(c)) > 3 else "")
                )
                display = df_al[["name", "position", "career_span", "teams", "coaches"]]
                display.columns = ["Player", "Pos", "Career", "Teams", "Coaches"]
                st.dataframe(display, use_container_width=True, hide_index=True)

                # Network view: show college alumni + their coaching connections
                if len(alumni) >= 2:
                    st.markdown("#### Coaching Network of Alumni")
                    st.caption(
                        "How did players from the same college end up in different (or the same) coaching systems?"
                    )
                    pids = [a["player_id"] for a in alumni if a["player_id"] in g.G]
                    if pids:
                        # Ego subgraph spanning all alumni + their coach neighbors
                        nodes = set(pids)
                        for pid in pids:
                            nodes.update(g.G.neighbors(pid))
                        subG = g.G.subgraph(nodes)
                        pos = nx.spring_layout(subG, seed=99, k=1.4)

                        ex2, ey2 = [], []
                        for u, v in subG.edges():
                            x0, y0 = pos[u]; x1, y1 = pos[v]
                            ex2 += [x0, x1, None]; ey2 += [y0, y1, None]

                        nx2, ny2, lbl2, col2, sz2 = [], [], [], [], []
                        for node in subG.nodes():
                            x, y = pos[node]
                            nx2.append(x); ny2.append(y)
                            data = subG.nodes[node]
                            lbl2.append(data.get("label", node))
                            if node in pids:
                                col2.append("#E31837"); sz2.append(18)
                            elif data.get("type") == "coach":
                                col2.append("#FFD700"); sz2.append(14)
                            else:
                                col2.append("#4a90d9"); sz2.append(9)

                        fig2 = go.Figure(data=[
                            go.Scatter(x=ex2, y=ey2, mode="lines",
                                       line=dict(width=1, color="#444"),
                                       hoverinfo="none", showlegend=False),
                            go.Scatter(x=nx2, y=ny2, mode="markers+text",
                                       text=lbl2, textposition="top center",
                                       marker=dict(size=sz2, color=col2,
                                                   line=dict(width=1, color="#222")),
                                       hoverinfo="text", showlegend=False),
                        ])
                        fig2.update_layout(
                            paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
                            font_color="#eee", height=420,
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            margin=dict(l=10, r=10, t=10, b=10),
                        )
                        st.caption("🔴 College alumni  🟡 Coaches  🔵 Other connected players")
                        st.plotly_chart(fig2, use_container_width=True)

                # Same-college connections for individual player
                st.markdown("#### Player → College Connections")
                pid_search = player_selectbox("See a specific player's college connections", "college_player")
                if pid_search:
                    same_college = g.same_college_connections(pid_search)
                    p_sel = g.players.get(pid_search)
                    if not same_college:
                        st.info(f"No other players from {p_sel.college if p_sel else '?'} in the dataset.")
                    else:
                        st.markdown(
                            f"**{p_sel.name if p_sel else pid_search}** ({p_sel.college if p_sel else ''}) "
                            f"shares a college with **{len(same_college)}** other NFL players:"
                        )
                        df_sc = pd.DataFrame(same_college)
                        df_sc["teams"] = df_sc["teams"].apply(lambda t: ", ".join(sorted(t)))
                        df_sc = df_sc[["name", "position", "career_span", "teams"]]
                        df_sc.columns = ["Player", "Pos", "Career", "NFL Teams"]
                        st.dataframe(df_sc, use_container_width=True, hide_index=True)
