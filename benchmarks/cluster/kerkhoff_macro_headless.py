"""Write an unattended copy of the Kerkhoff Live/Dead Fiji macro, for ``b_viability_external``.

``LDA_macro.ijm`` (Kerkhoff 2024, CC-BY-4.0; Zenodo 10395753) asks for its folders
with ``#@`` prompts and for its parameters in a dialog, so it cannot run without a
person at the screen. This script copies it with only that input replaced: the
folders and file suffix come from the macro argument ``"input,output,suffix"`` and
the parameters keep the macro's own defaults (prominence 20 and 20, live channel 2,
dead channel 1, Mean filter radius 0.5, cut prefix 17; the export format is csv).
Every processing line is the original.

    python kerkhoff_macro_headless.py LDA_macro.ijm LDA_headless.ijm

Run it as the macro of an ImageJ with Bio-Formats (Fiji's jars and plugins), under
a display (Bio-Formats builds its dialog even when given options), e.g.
``java -Xmx4g -cp 'Fiji/jars/*:Fiji/plugins/*' ij.ImageJ -port0 -macro
LDA_headless.ijm 'IN,OUT,tif'``; it writes ``OUT/Summary.csv`` (stop ImageJ after).
"""
from __future__ import annotations

import sys

START = "//Ask for the input folder, output folder, and file suffix"
END = "//Array of the Analysis Parameter Values for the Summary Data Table"
INPUT = """// UNATTENDED COPY of LDA_macro.ijm (Kerkhoff 2024, CC-BY-4.0; Zenodo 10395753),
// written by fluorostats benchmarks/cluster/kerkhoff_macro_headless.py. The only
// change: the "#@" folder prompts and the parameter dialog are replaced by the
// macro's own defaults (above) and the argument "input,output,suffix".
args = split(getArgument(), ",");
input = args[0];
output = args[1];
suffix = args[2];
form = "csv";
filt = "Mean";

"""


def main() -> None:
    src, dst = sys.argv[1:3]
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    start, end = text.index(START), text.index(END)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(text[:start] + INPUT + text[end:])


if __name__ == "__main__":
    main()
