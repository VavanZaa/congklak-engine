"""Convert the frozen opening report content to standalone LaTeX sources.

No search, training or source-data changes. Figures use native TikZ so the .tex
files can compile without additional project files in the Codex LaTeX editor.
The HTML/JSON content snapshot remains the input to this conversion only.
"""
from pathlib import Path
from html.parser import HTMLParser
import csv
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'reports/congklak-verified-openings-2026-10-17'
WORK = ROOT / 'build/report-latex-v1'
WORK.mkdir(parents=True, exist_ok=True)
BEFORE = WORK / 'before'
BEFORE.mkdir(exist_ok=True)
for name in ('Congklak-engine-report.pdf', 'YouTube-script-and-recording-plan.pdf',
             'manifest.json', 'START-HERE.txt', 'Congklak-report-and-video-kit.zip'):
    if not (BEFORE / name).exists():
        shutil.copy2(OUT / name, BEFORE / name)

CONTENT = json.loads((OUT / 'editable-content.json').read_text(encoding='utf-8'))
TRACE = json.loads((ROOT / 'build/report-openings-v1/seven-pit-trace.json').read_text())
SWEEP = json.loads((ROOT / 'build/report-openings-v1/sweep.json').read_text())

def esc(value):
    value = str(value)
    swaps = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
             '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
             '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
             '²': r'\(^{2}\)', '³': r'\(^{3}\)', '≥': r'\(\geq\)',
             '≤': r'\(\leq\)', '→': r'\(\to\)', '×': r'\(\times\)',
             '–': '--', '—': '---', '’': "'", '‘': "'", '“': '``', '”': "''",
             '\u00a0': '~'}
    return ''.join(swaps.get(c, c) for c in value)

class Inline(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
    def handle_starttag(self, tag, attrs):
        if tag in ('b', 'strong'): self.parts.append(r'\textbf{')
        elif tag in ('i', 'em'): self.parts.append(r'\emph{')
        elif tag in ('link', 'a'):
            url = dict(attrs).get('href', '')
            self.parts.append(r'\href{' + url.replace('%', r'\%').replace('#', r'\#') + '}{')
        elif tag == 'br': self.parts.append(r'\newline ')
        else: raise ValueError(tag)
    def handle_endtag(self, tag):
        if tag in ('b','strong','i','em','link','a'): self.parts.append('}')
    def handle_data(self, value): self.parts.append(esc(value))

def inline(value):
    p = Inline(); p.feed(str(value)); return ''.join(p.parts)

PREAMBLE = r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=20mm,top=18mm,bottom=20mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype}
\usepackage{amsmath,amssymb,booktabs,tabularx,array}
\usepackage[table]{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc}
\usepackage{fancyvrb,fancyhdr,titlesec}
\usepackage[hidelinks,unicode]{hyperref}
\definecolor{Ink}{HTML}{243B32}
\definecolor{Teal}{HTML}{E6F0EC}
\definecolor{Gold}{HTML}{DAB358}
\definecolor{Rule}{HTML}{D9DFDB}
\definecolor{Warm}{HTML}{F5F1E5}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\setlength{\headheight}{14pt}
\setlength{\emergencystretch}{3em}
\renewcommand{\arraystretch}{1.22}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\titleformat{\section}{\Large\bfseries\color{Ink}}{}{0pt}{}
\titleformat{\subsection}{\normalsize\bfseries\color{Ink}}{}{0pt}{}
\titlespacing*{\section}{0pt}{0pt}{12pt}
\titlespacing*{\subsection}{0pt}{9pt}{3pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize\sffamily CONGKLAK ENGINE PROJECT}
\fancyhead[R]{\footnotesize 17 October 2026}
\fancyfoot[L]{\footnotesize\color{Ink} Sequential opening \enspace / \enspace Single round}
\fancyfoot[R]{\footnotesize\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\fvset{fontsize=\footnotesize,baselinestretch=1.1,frame=single,framerule=0.3pt,rulecolor=\color{Rule}}
\hypersetup{pdftitle={TITLE},pdfauthor={Congklak engine project}}
\begin{document}
'''

def table(headers, rows, widths=None):
    n = len(headers)
    if widths:
        total = sum(widths)
        cols = ''.join(r'>{\hsize=' + f'{w*n/total:.6f}' + r'\hsize\linewidth=\hsize\raggedright\arraybackslash}X' for w in widths)
    else: cols = 'Y' * n
    lines = [r'\begingroup\small\setlength{\tabcolsep}{5pt}',
             r'\begin{tabularx}{\linewidth}{@{}' + cols + r'@{}}\toprule',
             r'\rowcolor{Teal}' + ' & '.join(r'\textbf{' + inline(x) + '}' for x in headers) + r' \\ \midrule']
    lines += [' & '.join(inline(x) for x in row) + r' \\' for row in rows]
    lines += [r'\bottomrule\end{tabularx}\endgroup', '']
    return '\n'.join(lines)

def flow_figure():
    return r'''\begin{center}
\begin{tikzpicture}[x=1cm,y=1cm,box/.style={draw=Ink!50,rounded corners=3pt,fill=Warm,align=center,text width=6.2cm,minimum height=1cm,font=\small},arr/.style={-{Latex},thick,draw=Ink}]
\node[box,fill=Teal,text width=13.4cm] (rules) at (0,0) {Frozen sequential rules and exact integer positions};
\node[box] (native) at (-3.7,-1.7) {General position analyzer\\Native C++20 CPU search};
\node[box] (finder) at (3.7,-1.7) {First-turn opening finder\\Python exact simulation and backtracking};
\node[box] (estimate) at (-3.7,-3.4) {Iterative deepening, TT and PVS\\Finite-horizon move estimates};
\node[box] (cert) at (3.7,-3.4) {Find one banked-majority witness\\Save a recoverable frontier and certificate};
\node[box,fill=Teal] (check) at (3.7,-5.1) {Separate literal verifier\\Replay each seed drop and rule transition};
\node[box] (ui) at (-3.7,-5.1) {Browser product\\Play, analyze and replay saved results};
\draw[arr] (rules.south) -- ++(0,-0.25) -| (native.north);
\draw[arr] (rules.south) -- ++(0,-0.25) -| (finder.north);
\draw[arr] (native) -- (estimate); \draw[arr] (estimate) -- (ui);
\draw[arr] (finder) -- (cert); \draw[arr] (cert) -- (check);
\draw[arr] (check) -- (ui);
\end{tikzpicture}
\end{center}'''

def board_figure():
    cells = TRACE[-1]['cells']
    lines = [r'\begin{center}\begin{tikzpicture}[x=0.93cm,y=1cm]',
             r'\fill[Warm,rounded corners=14pt] (-0.7,-0.9) rectangle (15.5,2.6);']
    for x, label, value, color in [(0,'A',cells[7],'Gold'),(14.8,'B',cells[15],'Teal')]:
        lines.append(r'\node[draw=Ink!40,fill='+color+r',rounded corners=12pt,minimum width=1.05cm,minimum height=2.6cm,align=center] at ('+str(x)+r',0.85) {\textbf{'+label+r'}\\[10pt]{\Large '+str(value)+r'}\\[10pt]\scriptsize STORE};')
    for side, y in [(1,1.7),(0,0)]:
        for col in range(7):
            pit = col if side else 6-col
            x=1.8+col*1.85
            lines.append(r'\node[circle,draw=Ink!35,fill=white,minimum size=0.85cm] at ('+str(x)+','+str(y)+r') {\large '+str(cells[side*8+pit])+r'};')
            lines.append(r'\node[font=\scriptsize] at ('+str(x)+','+str(y-0.64)+') {'+chr(65+side)+str(7-pit)+'};')
    lines += [r'\end{tikzpicture}\par\medskip',r'{\small A sows A7 $\to$ A6 $\to$ A5 $\to$ A4 $\to$ A3 $\to$ A2 $\to$ A1 $\to$ A store.}',r'\end{center}']
    return '\n'.join(lines)

def banked_figure():
    lines = [r'\begin{center}\begin{tikzpicture}[x=0.95cm,y=0.052cm]',
             r'\draw[->,Ink] (0,0) -- (13.7,0) node[right,font=\scriptsize] {Choice};',
             r'\draw[->,Ink] (0,0) -- (0,60) node[above,font=\scriptsize] {Banked seeds};']
    for n in range(0,51,10):
        lines.append(r'\draw[Rule] (0,'+str(n)+') -- (13,'+str(n)+');')
        lines.append(r'\node[left,font=\scriptsize] at (0,'+str(n)+') {'+str(n)+'};')
    lines += [r'\draw[dashed,red!65!black] (0,50) -- (13,50);',
              r'\node[anchor=south east,font=\scriptsize,text=red!65!black] at (12,50) {50 to win};']
    points = [(0,0)]+[(r['choice'],r['store']) for r in TRACE]
    lines.append(r'\draw[Ink,very thick] '+' -- '.join(f'({x},{y})' for x,y in points)+';')
    for x,y in points:
        lines.append(r'\fill[Gold] ('+str(x)+','+str(y)+r') circle (2pt);')
        lines.append(r'\node[below,font=\scriptsize] at ('+str(x)+',0) {'+str(x)+'};')
    lines += [r'\node[above right,font=\small\bfseries] at (13,54) {54};',r'\end{tikzpicture}\end{center}']
    return '\n'.join(lines)

def paired_results():
    rows = list(csv.DictReader((OUT/'results-n1-to-n64.csv').open(encoding='utf-8')))
    # Use recorded sweep values, not any recomputed search or timing.
    by_n = {r['n']:r for r in SWEEP['rows']}
    heads = ['n', 'Choices', 'Banked', 'Target', 'Tested']
    halves=[]
    for start in (1,33):
        data=[]
        for n in range(start,start+32):
            r=by_n[n]
            data.append([n,r['choices'] if r['choices'] is not None else '--',r['starter_store'] if r['starter_store'] is not None else '--',r['required'],r['simulated_actions']])
        halves.append(table(heads,data,[.35,.65,.65,.65,.65]))
    assert len(rows)==64
    return (r'\begin{minipage}[t]{0.485\linewidth}'+halves[0]+r'\end{minipage}\hfill'+
            r'\begin{minipage}[t]{0.485\linewidth}'+halves[1]+r'\end{minipage}')

def render(kind, name):
    title=CONTENT[kind][0]['title']
    lines=[PREAMBLE.replace('TITLE',esc(title))]
    for i, section in enumerate(CONTENT[kind]):
        if i: lines.append(r'\clearpage')
        lines.append(r'\section*{'+inline(section['title'])+'}')
        for block in section['blocks']:
            k=block['kind']
            if k=='h2': lines.append(r'\subsection*{'+inline(block['text'])+'}')
            elif k=='p': lines.append(inline(block['text'])+'\n')
            elif k=='small': lines.append(r'{\small '+inline(block['text'])+r'\par}'+'\n')
            elif k=='code': lines.append(r'\begin{Verbatim}'+'\n'+block['text']+'\n'+r'\end{Verbatim}')
            elif k=='table': lines.append(table(block['headers'],block['rows'],block.get('widths')))
            elif k=='paired-results': lines.append(paired_results())
            elif k=='image':
                figures={'engine-flow.png':flow_figure,'board-13.png':board_figure,'banked-seeds.png':banked_figure}
                lines.append(figures[block['file']]())
                lines.append(r'{\small\itshape '+inline(block['caption'])+r'\par}'+'\n')
            else: raise ValueError(k)
    lines.append(r'\end{document}')
    path=OUT/name
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'sections':len(CONTENT[kind])}

if __name__=='__main__':
    documents=[render('report','Congklak-engine-report.tex'),render('video','YouTube-script-and-recording-plan.tex')]
    receipt={'state':'sources_created','input_sha256':hashlib.sha256((OUT/'editable-content.json').read_bytes()).hexdigest(), 'documents':documents,'evidence_changed':False}
    (WORK/'status.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
