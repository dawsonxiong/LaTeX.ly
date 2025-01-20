import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

math_symbols = [
    r'0', r'1', r'2', r'3', r'4', r'5', r'6', r'7', r'8', r'9', r'+', r'-', r'=', r',',
    r'a', r'b', r'c', r'd', r'e', r'f', r'g', r'h', r'i', r'j', r'k', r'm', r'n',
    r'p', r'q', r'r', r's', r't', r'u', r'v', r'w', r'x', r'y', r'z',
    r'\{', r'\}', r'(', r')', r'!', r'|', r'<', r'>', r'/',
    r'\exists', r'\in', r'\forall', r'lim', r'\int', r'\infty', 
    r'\rightarrow', r'\pm', r'\times', r'\div', r'\geq', r'\leq', r'\neq',
    r'\alpha', r'\beta', r'\gamma', r'\delta', r'\theta', r'\lambda', r'\sigma', r'\mu', r'\pi', r'\Delta', r'\sum',
    r'\sin', r'\cos', r'\tan', r'log'
]
symbols = [
    r'0', r'1', r'2', r'3', r'4', r'5', r'6', r'7', r'8', r'9', r'+', r'-', r'=', r',',
    r'a', r'b', r'c', r'd', r'e', r'f', r'g', r'h', r'i', r'j', r'k', r'm', r'n',
    r'p', r'q', r'r', r's', r't', r'u', r'v', r'w', r'x', r'y', r'z',
    r'\{', r'\}', r'(', r')', r'!', r'|', r'<', r'>', r'/'
]

math_fonts = ["cm", "stix", "stixsans", "dejavuserif", "dejavusans"]
serif_fonts = ["Times New Roman", "Georgia", "Cambria", "Book Antiqua", "Palatino Linotype"]
sans_serif_fonts = ["Arial", "Calibri", "Verdana", "Segoe UI", "Century Gothic"]
monospace_fonts = ["Courier New", "Lucida Sans Typewriter", "OCR A Extended"]
cursive_fonts = ["Brush Script MT", "Lucida Handwriting", "Kristen ITC", "Freestyle Script", "Edwardian Script ITC"]
fantasy_fonts = ["Impact", "Chiller"]
fonts = {
    "serif": serif_fonts,
    "sans-serif": sans_serif_fonts,
    "monospace": monospace_fonts,
    "cursive": cursive_fonts,
    "fantasy": fantasy_fonts
}

# Dictionary for invalid file names
rename_map = {
    ">": "gt",
    "<": "lt",
    "|": "vertical-bar",
    "/": "forwards-slash"
}

# also wants to add support for fractions, exponents, degrees, sqrt, abs

# Set directory
base = r"test"
os.makedirs(base, exist_ok=True)


# Process math symbols
def process(symbol, dir):
    count = 0
    target_size = 64

    # Process math symbols
    for font in math_fonts:
        count += 1
        rcParams["mathtext.fontset"] = font

        plt.figure(figsize=(target_size / 100, target_size / 100), dpi=100)
        plt.text(0.5, 0.5, f"${symbol}$", fontsize=50, ha='center', va='center')

        ax = plt.gca()
        ax.set_position([0, 0, 1, 1])  # Remove excess space

        plt.axis('off')
        plt.tight_layout(pad=0)

        filename = str(count) + ".png"
        plt.savefig(os.path.join(dir, filename),
                    bbox_inches='tight',
                    pad_inches=0.01,
                    dpi=100,
                    format='png')
        plt.close()

    # Process non-math symbols
    if symbol in symbols:
        for font_type, font_list in fonts.items():
            for f in font_list:
                count += 1
                rcParams["font.family"] = font_type
                rcParams["font." + font_type] = [f]

                plt.figure(figsize=(target_size / 100, target_size / 100), dpi=100)
                ax = plt.gca()

                ax.set_xlim(-0.5, 1.5)
                ax.set_ylim(-0.5, 1.5)

                plt.text(0.5, 0.5, f"{symbol}",
                         fontsize=50, ha='center', va='center',
                         bbox=dict(boxstyle="square,pad=0", facecolor="white", edgecolor="none"))

                plt.axis('off')
                plt.tight_layout(pad=0)

                filename = str(count) + ".png"
                plt.savefig(os.path.join(dir, filename),
                            bbox_inches='tight',
                            pad_inches=0.01,
                            dpi=100,
                            format='png')
                plt.close()


# Process and save images
for symbol in math_symbols:
    sym = symbol.replace("\\", "")

    if sym in rename_map:
        sym = rename_map[sym]

    dir = os.path.join(base, sym)
    os.makedirs(dir, exist_ok=True)

    process(symbol, dir)
