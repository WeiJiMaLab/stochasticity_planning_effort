import matplotlib.pyplot as plt
import matplotlib
import os
import warnings

# Figures are typeset in Arimo, which is metrically compatible with Helvetica and
# Arial. The font files are NOT stored in this repository -- they are Google's to
# distribute under the SIL Open Font License, not ours. Run
# `bash analysis/scripts/get_fonts.sh` to download them into workflows/fonts/.
#
# If they are absent we fall back to whatever metric-compatible face is installed
# (Nimbus Sans ships with Ghostscript and most TeX installs; Liberation Sans with
# most Linux distributions). Last resort is matplotlib's DejaVu Sans, which is
# ~10% wider and can crowd tick labels in the narrower panels.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.normpath(os.path.join(_SRC_DIR, "..", "workflows", "fonts"))

FONT_STACK = ["Arimo", "Nimbus Sans", "Liberation Sans", "Helvetica", "Arial", "DejaVu Sans"]


def _resolve_font_family():
    """Return the font family to draw with, preferring the downloaded Arimo files.

    Registers workflows/fonts/*.ttf with matplotlib when present, then picks the
    first installed family in FONT_STACK. matplotlib substitutes DejaVu Sans for a
    missing family without saying so, so resolve explicitly and warn when only that
    last resort is available -- the substitution changes text metrics.
    """
    if os.path.isdir(_FONTS_DIR):
        for fname in sorted(os.listdir(_FONTS_DIR)):
            if fname.lower().endswith((".ttf", ".otf")):
                matplotlib.font_manager.fontManager.addfont(os.path.join(_FONTS_DIR, fname))

    installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for name in FONT_STACK:
        if name in installed:
            if name == FONT_STACK[-1]:
                warnings.warn(
                    f"None of {FONT_STACK[:-1]} are installed; falling back to {name}. "
                    "Figures will render, but text is wider than in the published "
                    "figures. Run analysis/scripts/get_fonts.sh to install Arimo.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return name
    return "sans-serif"


FONT_FAMILY = _resolve_font_family()
font_regular = matplotlib.font_manager.FontProperties(family=FONT_FAMILY, weight="normal")
font_bold = matplotlib.font_manager.FontProperties(family=FONT_FAMILY, weight="bold")


def set_text_style():
    """Set the Matplotlib default style and return the (regular, bold) FontProperties."""
    plt.style.use('default')
    matplotlib.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.size': 20
    })
    plt.rcParams['svg.fonttype'] = 'none'

    # Return both regular and bold FontProperties for convenience
    return font_regular, font_bold

set_text_style()

def draw_na_axes_panel(ax, *, alphabet_label=None, font_bold=None,
                       alphabet_xy=(-0.3, 1.15), alphabet_fontsize=28,
                       facecolor="#ebebeb",
                       panel_xy=(0.03, 0.12), panel_wh=(0.68, 0.76)):
    """Gray placeholder for axes where no model/data apply (e.g. EV baseline, volatility).

    ``axis('off')`` combined with axes ``facecolor`` alone rasterizes as white under the
    Agg backend; drawing an inset :class:`~matplotlib.patches.Rectangle` (transAxes)
    guarantees the background survives ``savefig``.

    ``panel_xy`` / ``panel_wh`` position a slightly undersized patch toward the left;
    ``N/A`` is centered on that rectangle.
    """
    from matplotlib.patches import Rectangle

    px, py = panel_xy
    pw, ph = panel_wh
    ax.add_patch(Rectangle((px, py), pw, ph, transform=ax.transAxes, facecolor=facecolor,
                           edgecolor="none", zorder=-1000, clip_on=False))
    ax.axis("off")
    ax.text(px + pw / 2, py + ph / 2, "N/A", transform=ax.transAxes, fontsize=22,
            va="center", ha="center", color="gray")
    if alphabet_label is not None and font_bold is not None:
        ax.text(*alphabet_xy, alphabet_label, transform=ax.transAxes, fontsize=alphabet_fontsize,
                fontproperties=font_bold, va="top", ha="left")

colormaps_ = {
    "blurple": ['#0c3679', '#807ad1', '#d067b9'],
    "political": ['#244b89', '#c6c9cd', '#b33756'],
    "popsicle": ['#ff5e57', '#ffbd69', '#2ec4b6'],
    "lavender": ['#240372', '#ecd5db', '#005ca3'],
    "sunset": ['#240372', '#d3033b', '#db7b51'],
    "arctic": ['#080745', '#246c99', '#92d9d4'],
    "easter": ['#eea79b', '#eca7dd', '#80b3ea', '#3cc3b3'],
    'countyfair': ['#f59e9e', '#a894d6', '#164374', '#2c9da5'],
    "playdough":['#d52320', '#2a78c0', '#06a288', '#fbae41'],
    "foliage": ['#d53f3f', '#ffc894', '#076e62'],
    "rouge": ['#6b0037', '#b40439', '#f2a673'],
    "berry": ['#64006b', '#b40462', '#f27373'],
    "sage": ['#043d48', '#196c2b', '#92a592'],
    "grass": ['#00663f', '#43896b', '#b9c997'],
}

colormaps = {name: plt.cm.colors.LinearSegmentedColormap.from_list(name, colors) for name, colors in colormaps_.items()}