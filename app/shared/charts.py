"""Reusable Plotly builders in the Dispatch theme."""
import plotly.graph_objects as go

BG = "#0a1520"
GRID = "rgba(255,255,255,0.05)"
FONT = "rgba(255,255,255,0.45)"

def base_layout(fig, height=250, pct_axis=True, yrange=None, legend=True):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Barlow, sans-serif", color=FONT, size=11),
        height=height,
        margin=dict(l=48, r=16, t=12, b=30),
        xaxis=dict(gridcolor=GRID, showline=False, zeroline=False),
        yaxis=dict(gridcolor=GRID, showline=False, zeroline=False),
        showlegend=legend,
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Mono, monospace", size=10)),
        hoverlabel=dict(bgcolor="#1b2d52", font_family="Barlow, sans-serif"),
    )
    if pct_axis:
        fig.update_yaxes(ticksuffix="%", tickformat=".0f")
    if yrange:
        fig.update_yaxes(range=yrange)
    return fig

def threshold_90(fig):
    fig.add_hline(y=90, line_dash="dot", line_color="rgba(239,68,68,0.35)",
                  annotation_text="90% threshold", annotation_position="top left",
                  annotation_font=dict(family="DM Mono, monospace", size=9, color="rgba(239,68,68,0.7)"))
    return fig
