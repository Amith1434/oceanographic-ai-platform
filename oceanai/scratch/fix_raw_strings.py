import re
from pathlib import Path

files = [
    Path('oceanai/app/Home.py'),
    Path('oceanai/app/pages/1_Ocean_Fishing_Sync.py'),
    Path('oceanai/app/pages/2_ML_Modeling.py'),
    Path('oceanai/app/pages/3_Ablation_Study.py'),
    Path('oceanai/app/pages/5_Anomaly_Detection.py'),
]

for fpath in files:
    if fpath.exists():
        content = fpath.read_text(encoding='utf-8')
        content = re.sub(r'(st\.(?:markdown|info|warning|write|latex))\("""', r'\1(r"""', content)
        fpath.write_text(content, encoding='utf-8')

print('Raw strings fixed in app files.')
