"""Create a draft-only source ZIP and independently verify its build."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

import fitz

ROOT = Path(__file__).resolve().parent
QA = ROOT / 'qa'
archive = ROOT / 'JEI_SOURCE_20261021.zip'
sources = [ROOT / name for name in ['README.md', 'AUTHOR_FIELDS_TO_COMPLETE_ZH.md', 'TEMPLATE_PROVENANCE_ZH.md', 'style_provenance.json', 'DOWNLOAD_SHA256.json']]
sources.append(ROOT / 'draw_review_figures.py')
sources.append(ROOT / 'build_review2_tables.py')
sources.extend(ROOT / name for name in ['draw_revision3_figures.py','redraw_fig2.py','build_revision3_results.py'])
sources.extend(path for path in (ROOT / 'review_evidence').rglob('*') if path.is_file())
superseded = {'Figure2.png', 'Figure5_quantitative_summary.pdf', 'Figure5_quantitative_summary.png'}
for folder in ['main', 'supplementary']:
    sources.extend(path for path in (ROOT / folder).rglob('*') if path.is_file() and path.name not in superseded and (path.suffix in {'.tex', '.bib', '.cls', '.bst'} or ('figures' in path.parts and path.suffix in {'.pdf', '.png','.svg','.json','.md','.csv'})))
manifest = {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
(QA / 'source_sha256.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in sources:
        bundle.write(path, path.relative_to(ROOT))
extracted = ROOT / 'independent_build_JEI_20261021_review3_delivery'
assert not extracted.exists(), 'Independent build directory must be fresh'
with zipfile.ZipFile(archive) as bundle:
    bundle.extractall(extracted)
result = {'archive': archive.name, 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'documents': {}}
for folder, stem in [('main', 'manuscript'), ('main', 'manuscript_reading'), ('supplementary', 'supplementary')]:
    key = 'reading' if stem == 'manuscript_reading' else folder
    commands = [['pdflatex', '-interaction=nonstopmode', '-halt-on-error', stem + '.tex']]
    if folder == 'main':
        commands.append(['bibtex', stem])
    commands += [['pdflatex', '-interaction=nonstopmode', '-halt-on-error', stem + '.tex']] * 2
    for index, command in enumerate(commands):
        proc = subprocess.run(command, cwd=extracted / folder, capture_output=True, text=True, encoding='utf-8', errors='replace')
        (QA / f'independent_{key}_{index + 1}.txt').write_text(proc.stdout + proc.stderr, encoding='utf-8')
        assert proc.returncode == 0, command
    original = fitz.open(ROOT / folder / (stem + '.pdf'))
    rebuilt = fitz.open(extracted / folder / (stem + '.pdf'))
    assert len(original) == len(rebuilt)
    equal = [hashlib.sha256(left.get_pixmap().samples).hexdigest() == hashlib.sha256(right.get_pixmap().samples).hexdigest() for left, right in zip(original, rebuilt)]
    assert all(equal), key + ': page pixel mismatch'
    result['documents'][key] = {'pages': len(original), 'independent_build_pass': True, 'all_page_pixels_identical_72dpi': True}
(QA / 'independent_build_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
