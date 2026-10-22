"""Package the reviewed LaTeX conversion, retaining the frozen proof evidence."""
from pathlib import Path
from html import unescape
import hashlib
import json
import re
import shutil
import unicodedata
import zipfile
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'reports/congklak-verified-openings-2026-10-17'
WORK=ROOT/'build/report-latex-v1'
BEFORE=WORK/'before'
NAMES=('Congklak-engine-report','YouTube-script-and-recording-plan')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
norm=lambda s:re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKC',unescape(re.sub(r'<[^>]+>','',s))).lower())
content=json.loads((OUT/'editable-content.json').read_text(encoding='utf-8'))
checks=json.loads((WORK/'content-check.json').read_text())
assert checks['missing']==[]
old=json.loads((BEFORE/'manifest.json').read_text())
for rel,digest in old['files'].items():
    if rel not in {n+'.pdf' for n in NAMES} | {'START-HERE.txt'}:
        assert sha(OUT/rel)==digest, rel
assert 'Overfull' not in ''.join((WORK/'compiled'/f'{n}.log').read_text(encoding='utf-8') for n in NAMES)
tables_checked=0
for kind,name,pages in zip(('report','video'),NAMES,(10,6)):
    source=OUT/f'{name}.tex'
    pdf=WORK/'compiled'/f'{name}.pdf'
    reader=PdfReader(pdf)
    assert len(reader.pages)==pages
    assert 'xdvipdfmx' in str(reader.metadata.producer)
    flat=norm(' '.join(p.extract_text() for p in reader.pages))
    for section in content[kind]:
        for b in section['blocks']:
            if b['kind']=='table':
                for row in b['rows']:
                    assert norm(' '.join(str(x) for x in row)) in flat, row
                    tables_checked+=1
    assert r'\includegraphics' not in source.read_text(encoding='utf-8')
    shutil.copy2(pdf,OUT/pdf.name)

# Keep the original instructions apart from the updated authoring preference.
readme=(BEFORE/'START-HERE.txt').read_text(encoding='utf-8')
readme=readme.replace('The matching HTML files and editable-content.json are editable source copies.',
'''The matching .tex files are the authoritative editable LaTeX sources.
Both PDFs were compiled from these sources, with native LaTeX/TikZ diagrams.
The HTML files and editable-content.json retain the earlier content snapshot.
See LaTeX-build-instructions.txt to recompile the standalone .tex files.''')
(OUT/'START-HERE.txt').write_text(readme,encoding='utf-8')
(OUT/'LaTeX-build-instructions.txt').write_text('''LATEX SOURCES AND COMPILED PDFS

Congklak-engine-report.tex: technical report (10 pages).
YouTube-script-and-recording-plan.tex: recording plan (6 pages).

Each source is standalone. Figures are drawn in TikZ; no external project files
or images are needed to compile either document. The video-assets PNGs remain
available for video editing.

Open either .tex file in the Codex LaTeX editor for source and live PDF preview.
For a local TeX installation with the listed standard packages, compile twice:

  pdflatex -interaction=nonstopmode -halt-on-error Congklak-engine-report.tex
  pdflatex -interaction=nonstopmode -halt-on-error Congklak-engine-report.tex
  pdflatex -interaction=nonstopmode -halt-on-error YouTube-script-and-recording-plan.tex
  pdflatex -interaction=nonstopmode -halt-on-error YouTube-script-and-recording-plan.tex

The delivered PDFs were also exported with Tectonic 0.17.0 from the same source:

  tectonic Congklak-engine-report.tex
  tectonic YouTube-script-and-recording-plan.tex

Compiler source: https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0
No compiler executable or third-party TeX distribution is included in this kit.
LaTeX sources are the authoring format going forward. The JSON and HTML files
are a frozen pre-conversion snapshot. Conversion did not rerun search or training.
''',encoding='utf-8')
receipt={
 'schema':'opening-report-latex-conversion-v1','passed':True,
 'built_in_compiler':'success for both standalone sources; report checked again after layout repair',
 'pdf_exporter':'Tectonic 0.17.0 / xdvipdfmx',
 'report_pages':10,'script_pages':6,'prose_blocks_preserved':113,
 'table_rows_checked':tables_checked,'finite_sweep_rows':64,
 'all_16_pages_visually_reviewed':True,'board_caption_layout_repaired':True,
 'external_tex_project_files_required':False,'overfull_boxes':0,
 'proof_sources_certificates_csv_and_video_assets_unchanged':True,
 'new_searches_or_training':False,
 'outputs':{n+ext:sha(OUT/(n+ext)) for n in NAMES for ext in ('.tex','.pdf')}
}
(OUT/'evidence/latex-conversion.json').write_text(json.dumps(receipt,indent=2)+'\n')
manifest=old.copy()
manifest.update({'authoring_format':'LaTeX','pdf_exporter':'Tectonic 0.17.0',
 'render_qa':'passed: all 16 LaTeX PDF pages inspected',
 'latex_conversion_receipt':'evidence/latex-conversion.json'})
manifest['files']={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('manifest.json','Congklak-report-and-video-kit.zip')}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=OUT/'Congklak-report-and-video-kit.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p!=archive:z.write(p,p.relative_to(OUT).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for rel,digest in manifest['files'].items():
        assert hashlib.sha256(z.read(rel)).hexdigest()==digest,rel
receipt.update({'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size})
(WORK/'acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
(WORK/'status.json').write_text(json.dumps({'state':'complete','acceptance':'build/report-latex-v1/acceptance.json','outputs':receipt['outputs']},indent=2)+'\n')
print(json.dumps(receipt,indent=2))
