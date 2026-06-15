"""
Foul-Won Analytics — EPL 2015/16
Predicting & exploring the probability a player wins a foul after he
receives or recovers the ball. StatsBomb open event data.

Single-file Streamlit app (st.tabs) following the HSR dashboard architecture:
Athletic-inspired light theme · Source Serif 4 / Inter / JetBrains Mono.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

# ----------------------------------------------------------------------------
# Page + theme
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Foul-Won Analytics · EPL 2015/16",
                   page_icon="\u26bd", layout="wide")

GREEN = "#1a6b3c"; INK = "#2b2b2b"; GREY = "#9aa0a6"
BG = "#f7f7f5"; CARD = "#ffffff"; ORANGE = "#d2682a"; LINE = "#e4e4df"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; color:{INK}; }}
.stApp {{ background:{BG}; }}
h1,h2,h3,h4 {{ font-family:'Source Serif 4',Georgia,serif !important; color:{INK} !important; letter-spacing:-0.01em; }}
.block-container {{ padding-top:2.2rem; max-width:1250px; }}
/* metric cards */
div[data-testid="stMetric"] {{
   background:{CARD}; border:1px solid {LINE}; border-radius:12px;
   padding:14px 18px; box-shadow:0 1px 2px rgba(0,0,0,0.03); }}
div[data-testid="stMetricValue"] {{ font-family:'JetBrains Mono',monospace; color:{GREEN}; font-weight:600; }}
div[data-testid="stMetricLabel"] {{ color:{GREY}; font-weight:500; }}
/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-family:'Source Serif 4',serif; font-size:1.02rem;
   font-weight:600; padding:8px 18px; color:{GREY}; }}
.stTabs [aria-selected="true"] {{ color:{GREEN} !important;
   border-bottom:2.5px solid {GREEN} !important; }}
.kicker {{ font-family:'JetBrains Mono',monospace; font-size:0.72rem; letter-spacing:0.14em;
   text-transform:uppercase; color:{GREEN}; font-weight:600; }}
.sub {{ color:{GREY}; font-size:0.95rem; margin-top:-6px; }}
.prob-big {{ font-family:'JetBrains Mono',monospace; font-size:3.6rem; font-weight:600;
   color:{GREEN}; line-height:1; }}
hr {{ border:none; border-top:1px solid {LINE}; margin:1.1rem 0; }}
</style>
""", unsafe_allow_html=True)

BASE_RATE = 0.0183  # league-wide foul-won-per-possession rate

def style_ax(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color(LINE)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(color=LINE, lw=0.8)
    ax.title.set_color(INK)
    return ax

def new_fig(w=7, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h)); fig.patch.set_facecolor(BG)
    return fig, style_ax(ax)

# ----------------------------------------------------------------------------
# Data + model
# ----------------------------------------------------------------------------
NUM = ['x','y','dist_to_goal','dist_to_own_goal','dist_to_touchline','central_dist',
       'minute','poss_seq','poss_passes_before','poss_time_elapsed','in_pass_length','pl_enc']
CAT = ['gain_type','pos_role','play_pattern','in_pass_height']
BOOL = ['under_pressure','on_poss_team','is_home_team','period_2nd']
FEATS = CAT + NUM + BOOL

@st.cache_data(show_spinner="Loading data\u2026")
def load_data():
    players = pd.read_csv("data/player_stats.csv")
    teams = pd.read_csv("data/team_stats.csv")
    gains = pd.read_parquet("data/gains.parquet")
    # normalise booleans coming back from csv
    for c in ['under_pressure']:
        gains[c] = gains[c].astype(str).str.lower().isin(['true','1'])
    gains['under_pressure'] = gains['under_pressure'].astype(int)
    # attach player foul-drawing tendency (pl_enc) to each touch
    enc = players.set_index('player_id')['pl_enc']
    gains['pl_enc'] = gains['player_id'].map(enc).fillna(BASE_RATE)
    return players, teams, gains

@st.cache_resource(show_spinner="Training model\u2026")
def get_model():
    _, _, gains = load_data()
    X = gains[FEATS].copy()
    X['in_pass_height'] = X['in_pass_height'].fillna('None')   # recoveries have no incoming pass
    y = gains['won_foul'].values
    pre = ColumnTransformer(
        [('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                                unknown_value=np.nan, encoded_missing_value=np.nan), CAT)],
        remainder='passthrough')
    pipe = Pipeline([('pre', pre),
                     ('hgb', HistGradientBoostingClassifier(
                         max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                         min_samples_leaf=200, l2_regularization=1.0,
                         early_stopping=True, random_state=42))])
    pipe.fit(X, y)
    return pipe

players, teams, gains = load_data()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<div class="kicker">English Premier League &middot; 2015/16 &middot; StatsBomb</div>',
            unsafe_allow_html=True)
st.markdown("# Foul-Won Analytics")
st.markdown('<p class="sub">Who draws fouls, where they happen, and the probability a foul '
            'is won on a given possession after a player receives or recovers the ball.</p>',
            unsafe_allow_html=True)
st.write("")

tab_player, tab_team, tab_pitch, tab_model = st.tabs(
    ["  Players  ", "  Teams  ", "  Pitch  ", "  Model  "])

ROLE_ORDER = ['FWD', 'MID', 'DEF', 'GK']

# ============================================================================
# PLAYER TAB
# ============================================================================
with tab_player:
    st.markdown("### Most-fouled players")
    st.markdown('<p class="sub">Fouls drawn across the season \u2014 totals, per-game rates, '
                'and breakdown by position.</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players", f"{len(players):,}")
    c2.metric("Total fouls drawn", f"{int(players['fouls_won'].sum()):,}")
    top = players.iloc[0]
    c3.metric("Most fouled", top['player_name'], f"{int(top['fouls_won'])} fouls")
    c4.metric("Highest per game",
              players.sort_values('fouls_per_game', ascending=False).iloc[0]['player_name'],
              f"{players['fouls_per_game'].max():.2f} / game")

    st.markdown("<hr>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns([1.1, 1.1, 1])
    metric = f1.radio("Rank by", ["Total fouls drawn", "Fouls drawn per game"], horizontal=True)
    role_opts = ["All positions"] + ROLE_ORDER
    role = f2.selectbox("Position group", role_opts)
    min_games = f3.slider("Minimum games", 1, 38, 5)

    sort_col = 'fouls_won' if metric.startswith("Total") else 'fouls_per_game'
    view = players[players['games'] >= min_games].copy()
    if role != "All positions":
        view = view[view['pos_role'] == role]
    view = view.sort_values(sort_col, ascending=False)

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(f"**Leaderboard \u2014 {metric.lower()}**")
        tbl = view.head(25)[['player_name', 'team', 'position', 'fouls_won',
                             'games', 'fouls_per_game']].reset_index(drop=True)
        tbl.index = tbl.index + 1
        st.dataframe(tbl, use_container_width=True, height=430,
            column_config={
                'player_name': st.column_config.TextColumn("Player"),
                'team': st.column_config.TextColumn("Team"),
                'position': st.column_config.TextColumn("Position"),
                'fouls_won': st.column_config.NumberColumn("Fouls drawn", format="%d"),
                'games': st.column_config.NumberColumn("Games", format="%d"),
                'fouls_per_game': st.column_config.NumberColumn("Per game", format="%.2f")})
    with right:
        st.markdown(f"**Top 12 \u2014 {metric.lower()}**")
        top12 = view.head(12).iloc[::-1]
        fig, ax = new_fig(5.4, 4.2)
        vals = top12[sort_col].values
        ax.barh(top12['player_name'], vals, color=GREEN)
        ax.set_xlabel(metric)
        for i, v in enumerate(vals):
            lbl = f"{int(v)}" if sort_col == 'fouls_won' else f"{v:.2f}"
            ax.text(v + max(vals)*0.01, i, lbl, va='center', fontsize=8.5)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Fouls drawn by position group**")
    pc1, pc2 = st.columns(2)
    grp = players.groupby('pos_role').agg(
        total=('fouls_won', 'sum'),
        per_game=('fouls_per_game', 'mean')).reindex(ROLE_ORDER).dropna()
    with pc1:
        fig, ax = new_fig(5.2, 3.0)
        ax.bar(grp.index, grp['total'], color=GREEN, width=0.6)
        for i, v in enumerate(grp['total']):
            ax.text(i, v + grp['total'].max()*0.01, f"{int(v):,}", ha='center', fontsize=9)
        ax.set_title("Total fouls drawn by position"); ax.set_ylabel("fouls")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with pc2:
        fig, ax = new_fig(5.2, 3.0)
        ax.bar(grp.index, grp['per_game'], color=ORANGE, width=0.6)
        for i, v in enumerate(grp['per_game']):
            ax.text(i, v + grp['per_game'].max()*0.01, f"{v:.2f}", ha='center', fontsize=9)
        ax.set_title("Avg fouls drawn per game (per player)"); ax.set_ylabel("per game")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ============================================================================
# TEAM TAB
# ============================================================================
with tab_team:
    st.markdown("### Team foul-drawing")
    st.markdown('<p class="sub">Click a row in the table to drill into a club and see its '
                'top three foul-winners.</p>', unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3)
    t1.metric("Teams", f"{len(teams)}")
    t2.metric("Total fouls drawn", f"{int(teams['fouls_won_total'].sum()):,}")
    t3.metric("Most fouled team", teams.iloc[0]['team'],
              f"{int(teams.iloc[0]['fouls_won_total'])} fouls")

    st.markdown("<hr>", unsafe_allow_html=True)
    tt = teams.copy()
    tt['foul_rate'] = (tt['fouls_won_total'] / tt['touches'] * 100).round(2)
    show = tt[['team', 'fouls_won_total', 'fouls_per_game', 'games', 'foul_rate']].reset_index(drop=True)

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**League table \u2014 fouls drawn**")
        event = st.dataframe(
            show, use_container_width=True, height=430, hide_index=True,
            on_select="rerun", selection_mode="single-row",
            column_config={
                'team': st.column_config.TextColumn("Team"),
                'fouls_won_total': st.column_config.NumberColumn("Fouls drawn", format="%d"),
                'fouls_per_game': st.column_config.NumberColumn("Per game", format="%.2f"),
                'games': st.column_config.NumberColumn("Games", format="%d"),
                'foul_rate': st.column_config.NumberColumn("Per 100 touches", format="%.2f")})
        sel = event.selection.rows
        default_team = show.iloc[sel[0]]['team'] if sel else show.iloc[0]['team']
    with right:
        st.markdown("**Drill into a club**")
        team_list = show['team'].tolist()
        chosen = st.selectbox("Team", team_list, index=team_list.index(default_team))
        row = tt[tt['team'] == chosen].iloc[0]
        m1, m2 = st.columns(2)
        m1.metric("Fouls drawn", f"{int(row['fouls_won_total'])}")
        m2.metric("Per game", f"{row['fouls_per_game']:.2f}")
        st.markdown(f"**Top 3 foul-winners \u2014 {chosen}**")
        top3 = (players[players['team'] == chosen]
                .sort_values('fouls_won', ascending=False)
                .head(3)[['player_name', 'position', 'fouls_won', 'fouls_per_game']]
                .reset_index(drop=True))
        top3.index = top3.index + 1
        st.dataframe(top3, use_container_width=True,
            column_config={
                'player_name': st.column_config.TextColumn("Player"),
                'position': st.column_config.TextColumn("Position"),
                'fouls_won': st.column_config.NumberColumn("Fouls drawn", format="%d"),
                'fouls_per_game': st.column_config.NumberColumn("Per game", format="%.2f")})

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Fouls drawn per game \u2014 all clubs**")
    fig, ax = new_fig(11, 3.6)
    order = tt.sort_values('fouls_per_game', ascending=False)
    bars = ax.bar(order['team'], order['fouls_per_game'], color=GREEN, width=0.66)
    if chosen in list(order['team']):
        bars[list(order['team']).index(chosen)].set_color(ORANGE)
    ax.set_ylabel("fouls drawn / game")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ============================================================================
# PITCH TAB
# ============================================================================
def draw_pitch(ax):
    ax.add_patch(Rectangle((0, 0), 120, 80, fill=False, ec=INK, lw=1.2))
    ax.add_patch(Rectangle((0, 18), 18, 44, fill=False, ec=INK, lw=1))
    ax.add_patch(Rectangle((102, 18), 18, 44, fill=False, ec=INK, lw=1))
    ax.add_patch(Rectangle((0, 30), 6, 20, fill=False, ec=INK, lw=1))
    ax.add_patch(Rectangle((114, 30), 6, 20, fill=False, ec=INK, lw=1))
    ax.plot([60, 60], [0, 80], color=INK, lw=1)
    ax.add_patch(Circle((60, 40), 10, fill=False, ec=INK, lw=1))
    ax.set_xlim(-3, 123); ax.set_ylim(-6, 83)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)

with tab_pitch:
    st.markdown("### Where fouls are won")
    st.markdown('<p class="sub">Foul-won rate per possession across the pitch, by play pattern, '
                'and by how the ball was gained. Direction of attack is left \u2192 right.</p>',
                unsafe_allow_html=True)

    flt = st.radio("Filter touches", ["All", "Receipts only", "Recoveries only"], horizontal=True)
    gp = gains
    if flt == "Receipts only":   gp = gains[gains['gain_type'] == 'receipt']
    elif flt == "Recoveries only": gp = gains[gains['gain_type'] == 'recovery']

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("**Foul-won rate by pitch location**")
        xb = np.linspace(0, 120, 13); yb = np.linspace(0, 80, 9)
        d = gp.copy()
        d['xbin'] = pd.cut(d['x'], xb); d['ybin'] = pd.cut(d['y'], yb)
        grid = d.groupby(['ybin', 'xbin'], observed=False)['won_foul'].mean().unstack() * 100
        fig, ax = new_fig(7.2, 4.8)
        im = ax.imshow(grid.values, origin='lower', extent=[0, 120, 0, 80],
                       aspect='equal', cmap='Greens', vmin=0, vmax=4)
        draw_pitch(ax)
        ax.annotate('', xy=(104, -3.5), xytext=(16, -3.5),
                    arrowprops=dict(arrowstyle='->', color=GREY))
        ax.text(60, -5.2, 'attacking direction', ha='center', color=GREY, fontsize=8)
        cb = plt.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
        cb.set_label('foul-won rate (%)', fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("Rate dips inside the box (penalty risk) and peaks through midfield.")
    with right:
        st.markdown("**By play pattern**")
        pp = (gp.groupby('play_pattern')['won_foul'].agg(['count', 'mean']))
        pp = pp[pp['count'] >= 500].sort_values('mean')
        fig, ax = new_fig(5.6, 2.6)
        cols = [ORANGE if i == 'From Counter' else GREEN for i in pp.index]
        ax.barh(pp.index, pp['mean']*100, color=cols)
        ax.axvline(BASE_RATE*100, color=GREY, ls='--', lw=1.1)
        for i, v in enumerate(pp['mean']*100):
            ax.text(v + 0.1, i, f"{v:.1f}", va='center', fontsize=8)
        ax.set_xlabel("foul-won rate (%)")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.markdown("**Receipt vs recovery**")
        gt = gains.groupby('gain_type')['won_foul'].mean()*100
        fig, ax = new_fig(5.6, 1.9)
        b = ax.bar(['Receipt', 'Recovery'],
                   [gt.get('receipt', 0), gt.get('recovery', 0)],
                   color=[GREEN, ORANGE], width=0.5)
        ax.axhline(BASE_RATE*100, color=GREY, ls='--', lw=1.1)
        for bar, v in zip(b, [gt.get('receipt', 0), gt.get('recovery', 0)]):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.05, f"{v:.2f}%", ha='center', fontsize=9)
        ax.set_ylabel("rate (%)")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("Recoveries lead to a won foul ~2.3\u00d7 as often as receipts.")

# ============================================================================
# MODEL TAB
# ============================================================================
with tab_model:
    st.markdown("### Foul-won probability calculator")
    st.markdown('<p class="sub">Set the situation and the model returns the probability the '
                'player wins a foul on this possession. Gradient-boosting model, well calibrated '
                '(predicted \u2248 observed across the range).</p>', unsafe_allow_html=True)

    model = get_model()
    pp_opts = sorted(gains['play_pattern'].dropna().unique().tolist())

    cfg, out = st.columns([1.25, 1])
    with cfg:
        a, b = st.columns(2)
        gain_type = a.radio("Ball gained by", ["receipt", "recovery"],
                            format_func=lambda s: s.capitalize())
        pos_role = b.selectbox("Player position", ROLE_ORDER, index=0)
        under_pressure = a.checkbox("Under pressure when gaining the ball", value=True)
        play_pattern = b.selectbox("Play pattern", pp_opts,
            index=pp_opts.index("From Counter") if "From Counter" in pp_opts else 0)

        st.markdown("**Player foul-drawing tendency**")
        names = ["League average"] + players.sort_values('fouls_won', ascending=False)['player_name'].tolist()
        who = st.selectbox("Use a specific player's profile (optional)", names)
        if who == "League average":
            pl_enc = BASE_RATE
        else:
            pl_enc = float(players.loc[players['player_name'] == who, 'pl_enc'].iloc[0])
        st.caption(f"Tendency used: {pl_enc*100:.2f}% (league avg {BASE_RATE*100:.2f}%)")

        st.markdown("**Location on pitch** (attack \u2192, x:0\u2013120, y:0\u201380)")
        lx, ly = st.columns(2)
        x = lx.slider("x \u2014 distance up pitch", 0, 120, 70)
        y = ly.slider("y \u2014 width", 0, 80, 40)

        st.markdown("**Possession context**")
        p1, p2, p3 = st.columns(3)
        passes_before = p1.slider("Passes so far", 0, 15, 0)
        time_elapsed = p2.slider("Seconds into possession", 0, 60, 3)
        minute = p3.slider("Match minute", 0, 95, 60)

        if gain_type == "receipt":
            ip1, ip2 = st.columns(2)
            in_pass_height = ip1.selectbox("Incoming pass height",
                                           ["Ground Pass", "Low Pass", "High Pass"])
            in_pass_length = ip2.slider("Incoming pass length (m)", 0, 60, 18)
        else:
            in_pass_height = "None"; in_pass_length = np.nan

    # assemble single-row feature frame
    row = {
        'gain_type': gain_type, 'pos_role': pos_role, 'play_pattern': play_pattern,
        'in_pass_height': in_pass_height,
        'x': x, 'y': y,
        'dist_to_goal': float(np.hypot(120 - x, 40 - y)),
        'dist_to_own_goal': float(np.hypot(x, 40 - y)),
        'dist_to_touchline': float(min(y, 80 - y)),
        'central_dist': float(abs(y - 40)),
        'minute': minute, 'poss_seq': passes_before, 'poss_passes_before': passes_before,
        'poss_time_elapsed': time_elapsed, 'in_pass_length': in_pass_length, 'pl_enc': pl_enc,
        'under_pressure': int(under_pressure), 'on_poss_team': 1,
        'is_home_team': 1, 'period_2nd': int(minute > 45)}
    Xrow = pd.DataFrame([row])[FEATS]
    prob = float(model.predict_proba(Xrow)[:, 1][0])
    lift = prob / BASE_RATE

    with out:
        st.markdown("#### Predicted probability")
        st.markdown(f'<div class="prob-big">{prob*100:.1f}%</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="sub">{lift:.1f}\u00d7 the league average of '
                    f'{BASE_RATE*100:.2f}% per possession</p>', unsafe_allow_html=True)
        # gauge-style bar vs reference points
        fig, ax = new_fig(5.0, 1.5)
        ax.barh([0], [prob*100], color=GREEN, height=0.5)
        ax.axvline(BASE_RATE*100, color=GREY, ls='--', lw=1.2)
        ax.text(BASE_RATE*100, 0.55, "league avg", color=GREY, fontsize=7.5, ha='center')
        ax.set_xlim(0, max(12, prob*100*1.25)); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_xlabel("probability (%)", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.caption("Biggest levers: under-pressure status, the player's foul-drawing "
                   "tendency, and counter-attacking play patterns. Try toggling 'under "
                   "pressure' or switching to a counter to see the swing.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f'<p class="sub">Data: StatsBomb open event data, EPL 2015/16 &middot; '
            f'323,322 ball receptions/recoveries &middot; league base rate {BASE_RATE*100:.2f}% '
            f'fouls won per possession.</p>', unsafe_allow_html=True)
