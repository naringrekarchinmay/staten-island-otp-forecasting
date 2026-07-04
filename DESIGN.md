# DESIGN.md

**Direction:** "Dispatch" — dark command-center. Source of truth: `design_handoff/README.md` (in the design bundle) + `SIR OTP Templates.dc.html`.

## Tokens
- Backgrounds: main `#0c1824`, sidebar `#0e1f30`, chart/card inner `#0a1520`
- Accent gold `#e9b820` (primary/selection/hero series), blue-teal `#4a8ab0` (secondary series), purple `#7b6caa` (tertiary series), navy `#1b2d52`
- Text: primary `#ffffff`, secondary `rgba(255,255,255,0.45)`, muted labels `#2e5a6e`/`#245060`
- Semantic: success `#22c55e`, danger `#ef4444`, warning `#f59e0b`
- Borders: `rgba(255,255,255,0.05–0.08)`; card radius 8px; chips 3–4px

## Type
- Display/headings/KPI values: Barlow Condensed 700/800, uppercase, 0.04em tracking (page title 38px, hero up to 80px)
- Body: Barlow 400/500, 12–15px
- Data labels/eyebrows: DM Mono 400/500, 8–11px, uppercase, 0.08–0.16em tracking (this is the handoff's deliberate kicker system, used as designed)

## Charts (Plotly)
- `paper_bgcolor` and `plot_bgcolor` `#0a1520`, grid `rgba(255,255,255,0.04)`, font `rgba(255,255,255,0.45)`
- Gold filled area for hero series (`rgba(233,184,32,0.15–0.25)` fill), dotted blue/purple secondaries, dotted red threshold lines at 90%

## Structure
- Entry `app/streamlit_app.py` + `app/views/*.py` via `st.navigation`; shared helpers in `app/shared/`
- Sidebar: gold SIR logo badge, uppercase nav, live-status footer
- KPI rows via styled `st.metric`; first metric gold-accented
