import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

math_symbols = [
    r'0', r'1', r'2', r'3', r'4', r'5', r'6', r'7', r'8', r'9', r'+', r'-', r'=', r',',
    r'a', r'b', r'c', r'd', r'e', r'f', r'g', r'h', r'i', r'j', r'k', r'l', r'm', r'n', 
    r'o', r'p', r'q', r'r', r's', r't', r'u', r'v', r'w', r'x', r'y', r'z', r'\{', r'\}', 
    r'(', r')', r'!', r'|', r'<', r'>', r'/',
    r'\supset', r'\supseteq', r'\nsupset', r'\subset', r'\subseteq', r'\nsubset', r'\setminus',
    r'\cap', r'\cup', r'\exists', r'\in', r'\notin',  r'\forall', r'\emptyset', 
    r'\propto',  r'\rightarrow', r'\leftarrow', r'\Rightarrow', r'\Leftarrow', r'\perp', r'\parallel',
    r'\partial', r'\int', r'\infty', r'\pm', r'\mp', r'\times', r'\div', 
    r'\therefore', r'\cdot', r'\equiv', r'\approx', r'\sim', r'\simeq', r'\geq',  r'\leq', r'\neq',
    r'\alpha', r'\beta', r'\gamma', r'\delta', r'\epsilon', r'\theta', r'\lambda', r'\tau', 
    r'\sigma', r'\phi', r'\omega', r'\mu', r'\pi', r'\Delta', r'\Pi', r'\Sigma', r'\Phi', r'\Omega',
    r'\sin', r'\cos', r'\tan', r'\cot', r'\csc', r'\sec'
]
symbols = [
    r'0', r'1', r'2', r'3', r'4', r'5', r'6', r'7', r'8', r'9', r'+', r'-', r'=', r',',
    r'a', r'b', r'c', r'd', r'e', r'f', r'g', r'h', r'i', r'j', r'k', r'l', r'm', r'n', 
    r'o', r'p', r'q', r'r', r's', r't', r'u', r'v', r'w', r'x', r'y', r'z', r'\{', r'\}', 
    r'(', r')', r'!', r'|', r'<', r'>', r'/'
]

math_fonts = ["cm", "stix", "stixsans", "dejavuserif", "dejavusans"]
serif_fonts = ["Times New Roman", "Georgia", "Cambria", "Book Antiqua", "Palatino Linotype" ]
sans_serif_fonts = ["Arial", "Calibri", "Verdana", "Segoe UI", "Century Gothic"]
monospace_fonts = ["Courier New", "Lucida Sans Typewriter", "OCR A Extended"]
cursive_fonts = ["Brush Script MT", "Lucida Handwriting", "Kristen ITC", "Freestyle Script", "Edwardian Script ITC"]
fantasy_fonts = ["Impact", "Chiller"]
fonts = {
    "serif" : serif_fonts, 
    "sans-serif" : sans_serif_fonts, 
    "monospace" : monospace_fonts, 
    "cursive" : cursive_fonts, 
    "fantasy" : fantasy_fonts
    }

# Dictionary to convert awkward symbol names to more readable names
rename_map = {
    "(": "left_bracket",
    ")": "right_bracket",
    ",": "comma",
    "-": "dash",
    "{": "left_curly",
    "}": "right_curly",
    "+": "plus",
    "=": "equals",
    "|": "abs_val",
    "<": "less_than",
    ">": "greater_than",
    "/": "slash"
}

# also wants to add support for fractions, exponents, degrees, sqrt, abs

# Set directory
base = r"data"
os.makedirs(base, exist_ok=True)

# Process math symbols
def process(symbol, dir):
    count = 0
    target_size = 32
    
    # Math symbols
    for font in math_fonts:
        count += 1
        rcParams["mathtext.fontset"] = font
        
        # Create figure with exact pixel size
        plt.figure(figsize=(target_size/100, target_size/100), dpi=100)
        
        plt.text(0.5, 0.5, f"${symbol}$", fontsize=20, ha='center', va='center')
        plt.axis('off')
        
        # Remove padding
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        filename = str(count) + ".png"
        plt.savefig(os.path.join(dir, filename), 
                   bbox_inches='tight',
                   pad_inches=0,
                   dpi=100,
                   format='png')
        plt.close()
    
    # Regular symbols
    if symbol in symbols:
        for font_type, font_list in fonts.items():
            for f in font_list:
                count += 1
                rcParams["font.family"] = font_type
                rcParams["font." + font_type] = [f]
                
                plt.figure(figsize=(target_size/100, target_size/100), dpi=100)
                
                plt.text(0.5, 0.5, f"{symbol}", fontsize=20, ha='center', va='center')
                plt.axis('off')
                
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                
                filename = str(count) + ".png"
                plt.savefig(os.path.join(dir, filename),
                          bbox_inches='tight',
                          pad_inches=0,
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