from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from os.path import join
import tempfile

from lib.utils.download_file import download_file

with tempfile.TemporaryDirectory() as tmpdir:
    download_file('https://example.org/', 'index.html', dir_prefix=tmpdir)
    filename = join(tmpdir, 'index.html')
    with open(filename, encoding='utf-8') as f:
        content = f.read()
        assert '<title>Example Domain</title>' in content
