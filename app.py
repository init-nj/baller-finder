import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

# from google import genai
#Doesn't work -Ran out of credits on the first day
# client = genai.Client(api_key="AIzaSyAZCYXKPt7mG6diaVhE-imbjEHpd_WHYO0")

# Doesn't work lads
# client = genai.Client(api_key="AIzaSyAZCYXKPt7mG6diaVhE-imbjEHpd_WHYO0")


st.set_page_config(page_title="Find Ballers", layout="wide", page_icon="🔍")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=DM+Mono:wght@400;500&family=Barlow:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }
.stApp { background: #0a0c10; color: #e8eaf0; }
.block-container { padding: 2rem 2.5rem; max-width: 1100px; }

.scout-header { font-family: 'Barlow Condensed', sans-serif; font-size: 28px;
  font-weight: 700; letter-spacing: 0.06em; color: #e8eaf0; }
.scout-header span { color: #00e5a0; }
.mono-label { font-family: 'DM Mono', monospace; font-size: 10px;
  color: #5a6070; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px; }
.stat-val { font-family: 'DM Mono', monospace; font-size: 22px;
  font-weight: 500; color: #00e5a0; }
.stat-label { font-family: 'DM Mono', monospace; font-size: 10px; color: #5a6070; }

.player-card { background: #111318; border: 1px solid #00e5a0;
  border-radius: 10px; padding: 16px; margin-bottom: 12px; }
.result-card { background: #111318; border: 1px solid #232730;
  border-radius: 10px; padding: 14px; margin-bottom: 8px; }
.low-data { background: #2a1f0a; color: #ba7517; font-size: 10px;
  padding: 2px 6px; border-radius: 4px; margin-left: 6px; }
.match-pct { font-family: 'DM Mono', monospace; font-size: 16px; color: #00e5a0; }
header[data-testid="stHeader"] {
    display: none !important;
}
.block-container {
    padding-top: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading player database...")
def load_data():
    HF_BASE = "https://huggingface.co/datasets/init-nj/ballers-dataset/resolve/main/data"

    appearances = pd.read_csv(f"{HF_BASE}/appearances.csv")
    players     = pd.read_csv(f"{HF_BASE}/players.csv")

    agg = appearances.groupby('player_id').agg(
        total_goals=('goals', 'sum'),
        total_assists=('assists', 'sum'),
        total_minutes=('minutes_played', 'sum'),
        total_appearances=('game_id', 'count'),
        total_yellows=('yellow_cards', 'sum'),
        total_reds=('red_cards', 'sum'),
    ).reset_index()

    agg['total_cards'] = agg['total_yellows'] + agg['total_reds']
    agg['mins_per_game'] = agg['total_minutes'] / agg['total_appearances']
    agg['low_confidence'] = agg['total_minutes'] < 200

    # Bayesian shrinkage
    weight = 500
    league_avg_goals   = (agg['total_goals'].sum()   / agg['total_minutes'].sum()) * 90
    league_avg_assists = (agg['total_assists'].sum() / agg['total_minutes'].sum()) * 90

    def shrunk(stat, mins, avg):
        return (stat + avg * (weight / 90)) / ((mins + weight) / 90)

    agg['goals_p90']   = agg.apply(lambda r: shrunk(r['total_goals'],   r['total_minutes'], league_avg_goals),   axis=1)
    agg['assists_p90'] = agg.apply(lambda r: shrunk(r['total_assists'], r['total_minutes'], league_avg_assists), axis=1)

    meta = players[['player_id','name','sub_position','country_of_citizenship','date_of_birth']].drop_duplicates('player_id')
    df = agg.merge(meta, on='player_id', how='left')
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], utc=True, errors='coerce')
    df['age'] = ((pd.Timestamp('today', tz='UTC') - df['date_of_birth']).dt.days / 365.25).fillna(0).astype(int)
    df['name'] = df['name'].fillna('Unknown')
    df['sub_position'] = df['sub_position'].fillna('Unknown')
    return df.reset_index(drop=True)


@st.cache_resource
def build_model(df):
    df = df.copy()
    df['position_encoded'] = LabelEncoder().fit_transform(df['sub_position'])
    feature_cols = ['goals_p90','assists_p90','total_cards','mins_per_game','age','position_encoded']
    X = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    knn = NearestNeighbors(n_neighbors=11, metric='cosine')
    knn.fit(X_scaled)
    return knn, X_scaled, scaler



'''def generate_scout_report(query_row, similar_df):
    # 1. Prepare data
    similar_summary = "\n".join([
        f"- {row['name']} ({row['sub_position']}, Age {row['age']}): "
        f"{row['goals_p90']:.2f} G/90, {row['assists_p90']:.2f} A/90, "
        f"{int(row['total_appearances'])} apps, {row['similarity']:.0f}% match"
        for _, row in similar_df.iterrows()
    ])

    prompt = f"""You are an elite football scout writing a concise professional report.

QUERY PLAYER:
Name: {query_row['name']}
Position: {query_row['sub_position']}
Nationality: {query_row['country_of_citizenship']}
Age: {query_row['age']}
Goals/90: {query_row['goals_p90']:.2f}
Assists/90: {query_row['assists_p90']:.2f}
Total appearances: {int(query_row['total_appearances'])}
Total minutes: {int(query_row['total_minutes'])}

SIMILAR PLAYERS FOUND BY THE RECOMMENDER:
{similar_summary}

Write a 3-paragraph scout report:
1. Profile the query player's statistical identity — what kind of player do the numbers suggest?
2. Justify why each similar player was recommended — what statistical traits do they share?
3. Flag anything notable — outliers, low-confidence players, age trajectories worth watching.

Be specific, cite the actual numbers, and write in the voice of a professional scout."""
    response = client.models.generate_content_stream(
    model='gemini-2.0-flash',
    contents=prompt
)

    for chunk in response:
        if chunk.text:
            yield chunk.text
'''

def make_radar(query_row, similar_df):
    cats = ['Goals/90','Assists/90','Cards/90','Mins/game','Age']
    
    fig = go.Figure()
    colors = ['#00e5a0','#5a8cff','#ff6b6b','#ffd166','#c77dff']

    plot_players = pd.concat([query_row.to_frame().T, similar_df.head(3)], ignore_index=True)
    all_data = load_data() 

    for i, (_, row) in enumerate(plot_players.iterrows()):
        # 1. Calculate values (Normalization)
        vals = [
            float((row['goals_p90']   - all_data['goals_p90'].min())   / (all_data['goals_p90'].max()   - all_data['goals_p90'].min()   + 1e-9)),
            float((row['assists_p90'] - all_data['assists_p90'].min()) / (all_data['assists_p90'].max() - all_data['assists_p90'].min() + 1e-9)),
            float((row['total_cards'] - all_data['total_cards'].min()) / (all_data['total_cards'].max() - all_data['total_cards'].min() + 1e-9)),
            float((row['mins_per_game'] - all_data['mins_per_game'].min()) / (all_data['mins_per_game'].max() - all_data['mins_per_game'].min() + 1e-9)),
            float((row['age'] - all_data['age'].min()) / (all_data['age'].max() - all_data['age'].min() + 1e-9)),
        ]

        # 2. Fix the NameError: Define rgba_color by converting hex to rgb
        hex_color = colors[i].lstrip('#')
        r, g, b = tuple(int(hex_color[j:j+2], 16) for j in (0, 2, 4))
        rgba_color = f'rgba({r}, {g}, {b}, 0.2)' # 0.2 alpha for transparency

        # 3. Add the trace
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],           # vals[0] closes the radar loop
            theta=cats + [cats[0]],       # cats[0] closes the radar loop
            fill='toself', 
            name=row['name'],
            line=dict(color=colors[i], width=2),
            fillcolor=rgba_color if i > 0 else colors[i],
            opacity=0.85 if i == 0 else 0.5,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor='#111318',
            radialaxis=dict(visible=True, range=[0,1], gridcolor='#232730', tickfont=dict(color='#5a6070', size=9)),
            angularaxis=dict(gridcolor='#232730', tickfont=dict(color='#8892a0', size=11)),
        ),
        paper_bgcolor='#111318', plot_bgcolor='#111318',
        font=dict(family='DM Mono, monospace', color='#e8eaf0'),
        legend=dict(bgcolor='#111318', bordercolor='#232730', borderwidth=1, font=dict(size=11)),
        margin=dict(l=40, r=40, t=30, b=30), height=380,
    )
    return fig


# ── App layout ────────────────────────────────────────────────────────────────

df = load_data()
knn, X_scaled, scaler = build_model(df)

st.markdown('<p class="scout-header">Baller-Finder<span>.</span>AI</p>', unsafe_allow_html=True)
st.markdown(f'<p class="mono-label">{len(df):,} players indexed</p>', unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    query = st.text_input("", placeholder="Search player name...", label_visibility="collapsed")
with col2:
    n_results = st.slider("Results", 3, 10, 5)
with col3:
    same_pos = st.toggle("Same position only", value=True)

if query:
    matches = df[df['name'].str.contains(query, case=False, na=False)]

    if matches.empty:
        st.warning(f"No player found for '{query}'")
    else:
        idx = matches.index[0]
        query_row = df.iloc[idx]

        distances, indices = knn.kneighbors([X_scaled[idx]], n_neighbors=n_results + 5)
        similar = df.iloc[indices[0][1:]].copy()
        similar['similarity'] = (1 - distances[0][1:]) * 100

        if same_pos:
            similar = similar[similar['sub_position'] == query_row['sub_position']]
        similar = similar.head(n_results)

        # Query player card
        st.markdown('<p class="mono-label" style="margin-top:1.5rem">Query player</p>', unsafe_allow_html=True)
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([3,1,1,1,1])
            with c1:
                st.markdown(f"""
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:20px;font-weight:700;color:#e8eaf0">{query_row['name']}</div>
                <div style="font-size:12px;color:#5a6070">{query_row['sub_position']} · {query_row['country_of_citizenship']} · Age {query_row['age']}</div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-val">{query_row["goals_p90"]:.2f}</div><div class="stat-label">G/90</div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="stat-val">{query_row["assists_p90"]:.2f}</div><div class="stat-label">A/90</div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="stat-val">{int(query_row["total_minutes"]):,}</div><div class="stat-label">MINS</div>', unsafe_allow_html=True)
            with c5:
                st.markdown(f'<div class="stat-val">{int(query_row["total_appearances"])}</div><div class="stat-label">APPS</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Results + Radar side by side
        left, right = st.columns([1, 1])

        with left:
            for rank, (_, row) in enumerate(similar.iterrows(), 1):
                low_data_badge = '<span class="low-data">low data</span>' if row['low_confidence'] else ''
                
                card_html = (
                    f'<div class="result-card">'
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
                    f'<div>'
                    f'<span style="font-family:DM Mono,monospace;font-size:10px;color:#5a6070">#{rank}</span>'
                    f'<span style="font-size:14px;font-weight:500;margin-left:8px">{row["name"]}</span>'
                    f'{low_data_badge}'
                    f'<div style="font-size:11px;color:#5a6070;margin-top:3px">'
                    f'{row["sub_position"]} · {row["country_of_citizenship"]} · Age {row["age"]}'
                    f'</div></div>'
                    f'<div class="match-pct">{row["similarity"]:.0f}%</div>'
                    f'</div>'
                    f'<div style="display:flex;gap:24px;margin-top:10px">'
                    f'<div><span style="font-family:DM Mono,monospace;font-size:13px">{row["goals_p90"]:.2f}</span> '
                    f'<span style="font-size:10px;color:#5a6070">G/90</span></div>'
                    f'<div><span style="font-family:DM Mono,monospace;font-size:13px">{row["assists_p90"]:.2f}</span> '
                    f'<span style="font-size:10px;color:#5a6070">A/90</span></div>'
                    f'<div><span style="font-family:DM Mono,monospace;font-size:13px">{int(row["total_appearances"])}</span> '
                    f'<span style="font-size:10px;color:#5a6070">APPS</span></div>'
                    f'</div></div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

        with right:
            st.markdown('<p class="mono-label">Radar comparison</p>', unsafe_allow_html=True)
            st.plotly_chart(make_radar(query_row, similar), use_container_width=True)
        '''
        Out of comission ~No Credits Left :(
                    Doesn't Work -> Ran out of Tokens!'
        st.markdown("---")
        st.markdown('<p class="mono-label">Scout report</p>', unsafe_allow_html=True)
        
        report_box = st.empty()
        
        with st.spinner(""):
            full_text = ""
            for chunk in generate_scout_report(query_row, similar):
                full_text += chunk
                report_box.markdown(f"""
                <div style="background:#111318;border:1px solid #232730;border-radius:10px;
                padding:20px 24px;font-size:14px;line-height:1.8;color:#c8cdd6;
                font-family:'Barlow',sans-serif;border-left:3px solid #00e5a0;">
                {full_text}▋
                </div>
                """, unsafe_allow_html=True)
        
        # Final render without cursor
        report_box.markdown(f"""
        <div style="background:#111318;border:1px solid #232730;border-radius:10px;
        padding:20px 24px;font-size:14px;line-height:1.8;color:#c8cdd6;
        font-family:'Barlow',sans-serif;border-left:3px solid #00e5a0;">
        {full_text}
        </div>
        """, unsafe_allow_html=True)
        '''
