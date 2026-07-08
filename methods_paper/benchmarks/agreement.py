"""Agreement plotting for method-comparison benchmarks.

The agreement *statistics* (Bland-Altman, Lin's CCC, ICC, agreement_report)
now live in the library — `fluorostats.agreement` — and are re-exported here
so existing benchmark scripts keep working. This module adds only the styled
two-panel figure (identity scatter + Bland-Altman) on top of them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Dogfood the library implementation rather than duplicating it.
from fluorostats.agreement import (  # noqa: F401
    bland_altman, lins_ccc, agreement_report,
    icc as icc_2way,
)


def plot_agreement(a, b, out_path, name_a="fluorostats", name_b="reference",
                   title=None, units=""):
    """Two-panel figure: identity-line scatter + Bland-Altman."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from fluorostats.style import apply_style, PALETTE
        apply_style()
        c_pt, c_line = PALETTE["accent"], PALETTE["ink"]
    except Exception:
        c_pt, c_line = "#E25C5C", "#1F2937"

    a = np.asarray(a, float); b = np.asarray(b, float)
    rep = agreement_report(a, b, name_a, name_b)
    ba = bland_altman(a, b)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Identity scatter
    lo = min(np.nanmin(a), np.nanmin(b)); hi = max(np.nanmax(a), np.nanmax(b))
    pad = (hi - lo) * 0.05 or 1
    ax1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color=c_line,
             lw=1, alpha=0.6, label="identity")
    ax1.scatter(b, a, s=55, color=c_pt, edgecolors=c_line, linewidths=0.6, alpha=0.85)
    ax1.set_xlabel(f"{name_b} {units}".strip())
    ax1.set_ylabel(f"{name_a} {units}".strip())
    ax1.set_title(f"Agreement  ·  CCC={rep['ccc']:.3f}  ICC={rep['icc']:.3f}  "
                  f"ρ={rep['spearman']:.3f}")
    ax1.legend(loc="upper left")

    # Bland-Altman
    ax2.scatter(ba["mean_vals"], ba["diff_vals"], s=55, color=c_pt,
                edgecolors=c_line, linewidths=0.6, alpha=0.85)
    ax2.axhline(ba["bias"], color=c_line, lw=1.4, label=f"bias={ba['bias']:.3g}")
    ax2.axhline(ba["loa_upper"], color=c_line, ls="--", lw=1, alpha=0.6,
                label=f"95% LoA [{ba['loa_lower']:.3g}, {ba['loa_upper']:.3g}]")
    ax2.axhline(ba["loa_lower"], color=c_line, ls="--", lw=1, alpha=0.6)
    ax2.set_xlabel(f"mean of methods {units}".strip())
    ax2.set_ylabel(f"{name_a} − {name_b}")
    ax2.set_title(f"Bland-Altman  (n={rep['n']})")
    ax2.legend(loc="best", fontsize=9)

    if title:
        fig.suptitle(title, fontweight="semibold")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return rep
