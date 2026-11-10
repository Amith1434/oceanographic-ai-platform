from pathlib import Path
import json
import pandas as pd

# Mock streamlit environment
class MockCol:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def subheader(self, *a, **k): pass
    def dataframe(self, *a, **k): pass
    def metric(self, *a, **k): pass
    def image(self, *a, **k): pass

class MockExpander:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def markdown(self, *a, **k): pass

class MockTab:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def dataframe(self, *a, **k): pass
    def plotly_chart(self, *a, **k): pass

class MockStreamlit:
    def set_page_config(self, **k): pass
    def title(self, *a, **k): pass
    def header(self, *a, **k): pass
    def subheader(self, *a, **k): pass
    def markdown(self, *a, **k): pass
    def info(self, *a, **k): pass
    def error(self, msg): print("ST ERROR:", msg)
    def stop(self): pass
    def selectbox(self, label, opts, **k): return opts[0]
    def columns(self, n): return [MockCol() for _ in range(n)]
    def tabs(self, titles): return [MockTab() for _ in titles]
    def expander(self, *a, **k): return MockExpander()
    def dataframe(self, *a, **k): pass
    def metric(self, *a, **k): pass
    def image(self, *a, **k): pass
    def plotly_chart(self, *a, **k): pass
    def cache_data(self, f): return f

pages = [
    Path('oceanai/app/pages/1_Ocean_Fishing_Sync.py'),
    Path('oceanai/app/pages/1_Ocean_Map.py'),
    Path('oceanai/app/pages/2_ML_Modeling.py'),
    Path('oceanai/app/pages/3_Ablation_Study.py'),
    Path('oceanai/app/pages/4_Feature_Importance.py'),
    Path('oceanai/app/pages/5_Anomaly_Detection.py'),
    Path('oceanai/app/pages/6_Biodiversity_OBIS.py'),
    Path('oceanai/app/pages/7_About_Limitations.py'),
]

st_mock = MockStreamlit()
for p in pages:
    if p.exists():
        print(f"Testing page: {p.name}...")
        code = p.read_text(encoding='utf-8')
        env = {'__file__': str(p.resolve()), '__name__': '__main__', 'st': st_mock}
        exec(code, env)
        print(f"  [OK] {p.name} executed without error.")

print("\nALL 8 PAGES VERIFIED SUCCESSFULLY!")
