import os
import sys

# make `import synth` and `import side_cut_vision` work when pytest runs from the package dir
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
