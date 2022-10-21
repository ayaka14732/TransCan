from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from lib.dataset.dummy import load_dummy

sentences = load_dummy()
assert isinstance(sentences, list)
assert isinstance(sentences[0], str)
