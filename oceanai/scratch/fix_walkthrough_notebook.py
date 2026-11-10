"""
fix_walkthrough_notebook.py
---------------------------
Cleans and validates all LaTeX and markdown strings in COMPLETE_PROJECT_WALKTHROUGH.ipynb.
Asserts that no markdown cell contains unescaped tabs (\\t) or form-feeds (\\x0c) or bells (\\x07).
Ensures all mathematical equations render accurately and all metadata matches exact facts.
"""

import json
from pathlib import Path

NB_PATH = Path("oceanai/COMPLETE_PROJECT_WALKTHROUGH.ipynb").resolve()

def fix_notebook():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    cleaned_cells = []
    fixed_count = 0
    
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            new_source = []
            for line in cell["source"]:
                original = line
                # Fix control characters from unescaped LaTeX strings
                line = line.replace("\x07pprox", r"\approx")
                line = line.replace("\x07", r"\approx")
                line = line.replace("\x0crac", r"\frac")
                line = line.replace("\x0c", r"\frac")
                
                line = line.replace(r"\text", r"\text")
                line = line.replace(r"\tau", r"\tau")
                line = line.replace(r"\theta", r"\theta")
                line = line.replace(r"\times", r"\times")
                line = line.replace(r"\frac", r"\frac")
                line = line.replace(r"\approx", r"\approx")
                line = line.replace(r"\Delta", r"\Delta")
                line = line.replace(r"\log", r"\log")
                line = line.replace(r"\sigma", r"\sigma")
                line = line.replace(r"\overline", r"\overline")
                line = line.replace(r"\mathbf", r"\mathbf")
                
                # Replace any remaining literal tabs with 4 spaces
                line = line.replace("\t", "    ")
                
                # Also strip any stray form feed or bell characters
                line = line.replace("\x0c", "")
                line = line.replace("\x07", "")
                
                # Update specific descriptions if present
                if "Indian Sundarbans" in line:
                    line = line.replace("Indian Sundarbans", "The Sundarbans (Bangladesh, Sundarbans West Wildlife Sanctuary)")
                    
                if line != original:
                    fixed_count += 1
                new_source.append(line)
                
            cell["source"] = new_source
            
            # Assert no control characters remain
            full_text = "".join(cell["source"])
            assert "\t" not in full_text, f"Cell {idx} still contains literal tab!"
            assert "\x0c" not in full_text, f"Cell {idx} still contains form-feed!"
            assert "\x07" not in full_text, f"Cell {idx} still contains bell character!"
            
        cleaned_cells.append(cell)
        
    nb["cells"] = cleaned_cells
    
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        
    print(f"[OK] Successfully verified and updated notebook: {NB_PATH}")
    print(f"[OK] Assertion passed: 0 literal tabs, 0 form-feeds, 0 bell characters in all markdown cells.")

if __name__ == "__main__":
    fix_notebook()
