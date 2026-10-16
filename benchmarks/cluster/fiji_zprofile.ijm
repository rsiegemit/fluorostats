// Fiji "Plot Z-axis Profile" of a z-stack, exported for producers/b_zprofile_fiji.py.
// Argument: "input_tif,output_csv". With no selection the profile is each slice's
// mean over the whole image; the plotted values are written with 9 decimals.
args = split(getArgument(), ",");
open(args[0]);
run("Set Measurements...", "mean decimal=9");
run("Plot Z-axis Profile");
Plot.getValues(x, y);
f = File.open(args[1]);
print(f, "slice,fiji_mean");
for (i = 0; i < y.length; i++) print(f, "" + (i + 1) + "," + d2s(y[i], 9));
File.close(f);
run("Quit");
