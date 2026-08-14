import matplotlib.pyplot as plt

POINT_COLOR = "black"
SMOOTH_COLOR = "#d62728"
REFERENCE_COLOR = "gray"
OK_COLOR = "#228B22"
NOT_OK_COLOR = "#cc0000"
CHECK_COLOR = "darkorange"
DEFAULT_MARKER_SIZE = 18


def apply_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.size": 10
    })