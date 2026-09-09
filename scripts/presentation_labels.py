"""Readable French method names, colours and the configuration mapping table.

Internal configuration IDs stay out of the figures; they live in the mapping
table written next to them (``mapping_configurations.csv`` / ``.md``).
"""
from __future__ import annotations

BASELINE = "R32__Gnone__input_bilinear__all19"

# resolution schedule -> readable name
RES_FR = {"R32": "32 constant",
          "Rprog": "Résolution 16→24→32",
          "Rgentle": "Résolution 24→32",
          "Rreverse": "Ordre inversé 24→16→32"}

GAUSS_FR = {"Gnone": "sans filtre",
            "Gplateau": "Gaussian par paliers",
            "Ggeo": "Gaussian géométrique",
            "Gmix": "Mélange identité–Gaussian"}

RED_FR = {"input_bilinear": "Bilinéaire sur l'image",
          "input_max": "Max sur l'image",
          "stem_bilinear": "Bilinéaire après la première convolution",
          "stem_max": "Max après la première convolution"}

MASK_FR = {"all19": "19 insertions", "early7": "Gaussian — 7 insertions"}

# the five methods of the main overview, in legend order, with fixed colours
MAIN = [
    (BASELINE, "Témoin — 32 constant, sans filtre", "#000000", 3.0, "-"),
    ("R32__Gplateau__input_bilinear__all19",
     "Gaussian par paliers (32 constant)", "#1f77b4", 2.0, "-"),
    ("Rprog__Gnone__input_bilinear__all19",
     "Résolution 16→24→32", "#2ca02c", 2.0, "-"),
    ("Rprog__Gplateau__input_bilinear__all19",
     "Résolution + Gaussian", "#d62728", 2.0, "-"),
    ("Rprog__Gnone__stem_max__all19",
     "Max après la première convolution", "#9467bd", 2.0, "-"),
]
MAIN_COLOR = {cid: c for cid, _l, c, _w, _s in MAIN}
MAIN_LABEL = {cid: l for cid, l, _c, _w, _s in MAIN}

# consistent colours for the focused figures
GAUSS_COLOR = {"Gnone": "#7a7a7a", "Gplateau": "#1f77b4",
               "Ggeo": "#2ca02c", "Gmix": "#d62728"}
RED_COLOR = {"input_bilinear": "#1f77b4", "input_max": "#ff7f0e",
             "stem_bilinear": "#2ca02c", "stem_max": "#9467bd"}
RES_COLOR = {"R32": "#7a7a7a", "Rprog": "#d62728",
             "Rgentle": "#1f77b4", "Rreverse": "#ff7f0e"}

FAMILY_FR = {
    "A": "Grille principale — résolution × calendrier gaussien",
    "B": "Mélange identité–Gaussian",
    "C": "Nombre d'insertions du filtre",
    "D": "Opérateur et emplacement de la réduction",
    "E": "Ordre des résolutions",
}

SCHEDULE_NOTE = (
    "Transitions de résolution après 6 et 12 époques complétées ; "
    "le filtrage cesse à l'entraînement à partir de l'époque 21 (indice zéro)."
)

SD_NOTE = ("Bandes / barres = ± 1 écart-type d'échantillon sur 3 graines "
           "(descriptif, ce ne sont pas des intervalles de confiance).")


def label_for(cid: str, style: str = "full") -> str:
    """A readable French label for a configuration id."""
    if cid == BASELINE and style != "bare":
        return "Témoin — 32 constant, sans filtre"
    res, g, red, mask = cid.split("__")
    bits = [RES_FR[res]]
    if g != "Gnone":
        bits.append(GAUSS_FR[g])
    else:
        bits.append("sans filtre")
    if red != "input_bilinear":
        bits.append(RED_FR[red])
    if mask != "all19":
        bits.append(MASK_FR[mask])
    return ", ".join(bits)


PLATEAU_TXT = "1,00 / 0,85 / 0,70 / 0,60 / 0,50 / 0,40 / 0,30 par blocs de 3 époques, puis 0"
GEO_TXT = "sigma = 0,9^e jusqu'à l'époque 20, puis 0"
MIX_TXT = "alpha = mêmes paliers ; T_alpha(h) = (1-alpha)h + alpha G(h), puis alpha = 0"


def mapping_rows(configs: dict) -> list:
    """Rows for the configuration mapping table."""
    rows = []
    for cid, c in sorted(configs.items()):
        res, g, red, mask = cid.split("__")
        detail = {"Gplateau": PLATEAU_TXT, "Ggeo": GEO_TXT,
                  "Gmix": MIX_TXT}.get(g, "aucun filtre interne")
        rows.append({
            "id_configuration": cid,
            "famille": FAMILY_FR[c["group"]],
            "nom_lisible": label_for(cid),
            "resolution": RES_FR[res],
            "calendrier_filtre": GAUSS_FR[g],
            "detail_calendrier": detail,
            "reduction": RED_FR[red],
            "insertions": "19 (tronc + 2 par bloc)" if mask == "all19"
                          else "7 (tronc + étage 1)",
        })
    return rows
