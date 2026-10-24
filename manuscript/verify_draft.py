"""Build and inspect the JEI draft against preserved pre-revision tables."""
from pathlib import Path
import json
import re
import subprocess

import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT.parent / 'original/manuscript_before_revision'
QA = ROOT / 'qa'
QA.mkdir(exist_ok=True)
TARGETS = [('main', 'manuscript'), ('main', 'manuscript_reading'), ('supplementary', 'supplementary')]


def run(args, cwd, log):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    log.write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'{args} failed; see {log}')


def build(base, prefix):
    for folder, stem in TARGETS:
        cwd = base / folder
        run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', stem + '.tex'], cwd, QA / f'{prefix}_{stem}_pass1.txt')
        if folder == 'main':
            run(['bibtex', stem], cwd, QA / f'{prefix}_{stem}_bibtex.txt')
        for count in [2, 3]:
            run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', stem + '.tex'], cwd, QA / f'{prefix}_{stem}_pass{count}.txt')


def numeric_tables(tex, parameters=False):
    tables = re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', tex, re.S)
    if parameters:
        return [[line.split('&')[3].strip() for line in table.splitlines() if len(line.split('&')) == 5 and not line.startswith('\\textbf')] for table in tables]
    return [re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?', table) for table in tables]


def render(base):
    images = []
    for old in QA.glob('contact_*.png'):
        old.unlink()
    for folder, stem in TARGETS:
        out = QA / stem
        out.mkdir(exist_ok=True)
        for old in out.glob('page-*.png'):
            old.unlink()
        run(['pdftoppm', '-r', '110', '-png', str(base / folder / (stem + '.pdf')), str(out / 'page')], ROOT, QA / f'{stem}_render.txt')
        images.extend(sorted(out.glob('page-*.png')))
    for start in range(0, len(images), 4):
        batch = images[start:start + 4]
        sheet = Image.new('RGB', (1360, 1840), 'white')
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(batch):
            thumb = Image.open(path).convert('RGB')
            thumb.thumbnail((660, 870))
            x, y = (index % 2) * 680, (index // 2) * 920
            draw.text((x + 8, y + 6), str(path.relative_to(QA)), fill='black')
            sheet.paste(thumb, (x, y + 28))
        sheet.save(QA / f'contact_{start // 4 + 1:02d}.png')


build(ROOT, 'draft')
summary = {'status': 'JEI_REVIEW3_UNIFIED_CURRENT_RESULTS_AND_ARCHIVED_REFERENCE', 'documents': {}}
for folder, stem in TARGETS:
    shared_stem = 'manuscript' if folder == 'main' else stem
    current = (ROOT / folder / (shared_stem + '.tex')).read_text(encoding='utf-8')
    previous = (OFFICIAL / folder / (shared_stem + '.tex')).read_text(encoding='utf-8')
    preserved = ((ROOT / 'supplementary/legacy_reference_tables.tex').read_text(encoding='utf-8')
                 if folder == 'main' else current)
    assert numeric_tables(preserved, folder == 'supplementary') == numeric_tables(previous, folder == 'supplementary'), folder + ': preserved numerical table values changed'
    run(['pdffonts', str(ROOT / folder / (stem + '.pdf'))], ROOT, QA / f'{stem}_fonts.txt')
    fonts = re.findall(r'\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$', (QA / f'{stem}_fonts.txt').read_text(), re.M)
    assert fonts and all(row[0] == 'yes' for row in fonts), stem + ': unembedded fonts'
    log = (ROOT / folder / (stem + '.log')).read_text(encoding='utf-8', errors='replace')
    summary['documents'][stem] = {
        'pages': len(fitz.open(ROOT / folder / (stem + '.pdf'))),
        'original_numerical_table_tokens_unchanged': True,
        'preserved_table_count': len(numeric_tables(previous, folder == 'supplementary')),
        'all_fonts_embedded': True,
        'warnings': re.findall(r'(?:LaTeX|Package [^\n]+) Warning:[^\n]*(?:\n[^\n]*)?', log),
        'overfull_boxes': re.findall(r'Overfull[^\n]*', log),
        'underfull_boxes': re.findall(r'Underfull[^\n]*', log),
        'unresolved': re.findall(r'[^\n]*(?:undefined|multiply defined|Rerun to get)[^\n]*', log),
    }
render(ROOT)
(QA / 'build_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
