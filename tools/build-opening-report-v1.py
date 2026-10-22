"""Build the evidence-based report, narration, editable HTML and video figures."""
from pathlib import Path
import csv
import hashlib
import html
import json
import shutil
import statistics
import re
import textwrap
import zipfile

from PIL import Image,ImageDraw,ImageFont
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT,TA_CENTER
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image as PDFImage,Preformatted,KeepTogether
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parent.parent
EVIDENCE=ROOT/'build/report-openings-v1'
OUT=ROOT/'reports/congklak-verified-openings-2026-10-17'
OUT.mkdir(parents=True,exist_ok=True)
ASSETS=OUT/'video-assets';ASSETS.mkdir(exist_ok=True)
SWEEP=json.loads((EVIDENCE/'sweep.json').read_text())
AUDIT=json.loads((EVIDENCE/'independent-verification.json').read_text())
TRACE=json.loads((EVIDENCE/'seven-pit-trace.json').read_text())
ROWS=SWEEP['rows']
assert AUDIT['passed'] and AUDIT['verified_count']==63

FONT=Path('C:/Windows/Fonts')
for name,file in [('Body','arial.ttf'),('BodyBold','arialbd.ttf'),('Display','georgia.ttf'),('Mono','consola.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONT/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
BLACK=colors.black;NAVY=colors.HexColor('#263b47');GRAY=colors.HexColor('#d9d9d9')
styles={
 'title':ParagraphStyle('title',fontName='Display',fontSize=27,leading=32,textColor=BLACK,spaceAfter=15),
 'h1':ParagraphStyle('h1',fontName='Display',fontSize=20,leading=25,textColor=BLACK,spaceAfter=13),
 'h2':ParagraphStyle('h2',fontName='BodyBold',fontSize=11.5,leading=15,textColor=BLACK,spaceBefore=13,spaceAfter=7),
 'p':ParagraphStyle('p',fontName='Body',fontSize=10.2,leading=14.9,textColor=BLACK,spaceAfter=9),
 'small':ParagraphStyle('small',fontName='Body',fontSize=8.5,leading=11.7,textColor=BLACK,spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=8.8,leading=11.5,textColor=BLACK),
 'th':ParagraphStyle('th',fontName='BodyBold',fontSize=8.8,leading=11.5,textColor=colors.white),
 'code':ParagraphStyle('code',fontName='Mono',fontSize=8.1,leading=11.3,textColor=BLACK,spaceAfter=10),
 'caption':ParagraphStyle('caption',fontName='Body',fontSize=8.5,leading=11.5,textColor=BLACK,spaceAfter=9),
}

def face(size,bold=False):return ImageFont.truetype(str(FONT/('arialbd.ttf' if bold else 'arial.ttf')),size)
PAPER='#f8f6ef';INK='#243b32';AMBER='#dab358';TEAL='#81b8a7'
def centered(draw,xy,text,font,fill=INK):
    box=draw.textbbox((0,0),str(text),font=font);draw.text((xy[0]-(box[2]-box[0])/2,xy[1]-(box[3]-box[1])/2),str(text),font=font,fill=fill)

def board_frame(index):
    im=Image.new('RGB',(1920,1080),PAPER);d=ImageDraw.Draw(im)
    cells=([7]*7+[0]+[7]*7+[0]) if index==0 else TRACE[index-1]['cells']
    title='Seven pits with seven seeds each' if index==0 else f'Choice {index} of 13   A{TRACE[index-1]["display_pit"]}'
    d.text((95,58),title,font=face(55,True),fill=INK)
    d.text((98,140),'Sequential opening   Single round   Store on your left',font=face(30),fill='#64776a')
    d.rounded_rectangle((95,255,1825,810),radius=95,fill='#e8e0c9',outline='#d0c8b1',width=3)
    for side,x,color in [(0,128,AMBER),(1,1674,TEAL)]:
        d.rounded_rectangle((x,300,x+118,765),radius=55,fill=color)
        centered(d,(x+59,395),chr(65+side),face(32,True));centered(d,(x+59,515),cells[7 if side==0 else 15],face(58,True))
        centered(d,(x+59,630),'STORE',face(20,True))
    for side,y in [(1,420),(0,650)]:
        for display in range(7):
            pit=display if side==1 else 6-display;x=350+display*198
            chosen=index>0 and side==0 and pit==TRACE[index-1]['canonical_action']
            d.ellipse((x-80,y-80,x+80,y+80),fill='#fffdf3',outline='#a88639' if chosen else '#cfc7b0',width=6 if chosen else 2)
            centered(d,(x,y-5),cells[side*8+pit],face(51))
            centered(d,(x,y+108),f'{chr(65+side)}{7-pit}',face(24,True))
    d.text((98,860),'A sows A7 > A6 > A5 > A4 > A3 > A2 > A1 > A store',font=face(28),fill=INK)
    if index==0:detail='The starter chooses A7 first. Each displayed step completes all relays in that choice.'
    else:
        row=TRACE[index-1]
        ending='Capture ends the turn.' if row['reason']=='own_capture' else 'Own-store landing gives another choice.'
        detail=f'{row["relays"]} relays   {row["deposits"]} seed drops   A store {row["store"]}   {ending}'
    d.text((98,916),detail,font=face(27),fill=INK)
    if index==13:d.text((98,985),'54 secured for A   |   at most 44 for B   |   no opponent choice before the target',font=face(30,True),fill=INK)
    else:d.text((98,990),'Exact counts and rules define the guarantee. This image is a stable replay position.',font=face(23),fill='#64776a')
    im.save(ASSETS/f'board-{index:02d}.png')
for i in range(14):board_frame(i)

def progress_figure():
    im=Image.new('RGB',(1920,1080),PAPER);d=ImageDraw.Draw(im)
    d.text((100,65),'Banking a guaranteed majority',font=face(61,True),fill=INK)
    d.text((104,160),'Seven pits   98 seeds total   Winning threshold 50',font=face(34),fill=INK)
    x0,y0,x1,y1=145,280,1770,790
    for value in (0,10,20,30,40,50):
        y=y1-value/56*(y1-y0);d.line((x0,y,x1,y),fill='#dedfd5',width=2)
        d.text((88,y-15),str(value),font=face(23),fill='#64776a')
    target_y=y1-50/56*(y1-y0);d.line((x0,target_y,x1,target_y),fill='#a94e38',width=4)
    d.text((1470,target_y-44),'50 to win',font=face(25,True),fill='#a94e38')
    points=[(x0,y1)]+[(x0+r['choice']/13*(x1-x0),y1-r['store']/56*(y1-y0)) for r in TRACE]
    d.line(points,fill=INK,width=7)
    for i,(x,y) in enumerate(points):
        d.ellipse((x-8,y-8,x+8,y+8),fill=AMBER);centered(d,(x,y1+35),str(i),face(23))
    d.text((800,875),'Completed pit choice',font=face(28),fill=INK)
    d.text((100,975),'Choice 12 banks 49. Choice 13 banks 54 and ends with a capture.',font=face(35,True),fill=INK)
    im.save(ASSETS/'banked-seeds.png')
progress_figure()

def coverage_figure():
    im=Image.new('RGB',(1920,1080),PAPER);d=ImageDraw.Draw(im)
    d.text((100,55),'A finite sweep from n 1 to n 64',font=face(60,True),fill=INK)
    d.text((105,145),'63 independently verified majority-winning witnesses',font=face(35),fill=INK)
    for n in range(1,65):
        row,col=divmod(n-1,8);x=112+col*215;y=245+row*79
        d.rounded_rectangle((x,y,x+185,y+62),radius=8,fill='#e2e3db' if n==1 else '#d8e8dc',outline='#b9c5b5',width=2)
        centered(d,(x+92,y+30),str(n),face(28,True))
    d.text((110,935),'n = 1: first-turn search exhausted; its only opening choice was checked separately.',font=face(28),fill=INK)
    d.text((110,991),'Verified for n = 2 through 64 under our rules. No theorem for arbitrary n is claimed.',font=face(28,True),fill=INK)
    im.save(ASSETS/'coverage.png')
coverage_figure()

def architecture_figure():
    im=Image.new('RGB',(1920,1080),PAPER);d=ImageDraw.Draw(im)
    d.text((100,70),'The engine and the proof checker',font=face(60,True),fill=INK)
    boxes=[(110,260,870,440,'Position analyzer','C++ simulator + alpha-beta + TT + PVS'),
           (110,550,870,730,'Opening finder','Python exact sowing + first-witness DFS'),
           (1070,550,1810,730,'Independent verifier','Literal seed-by-seed replay'),
           (1070,815,1810,965,'Winning certificate','Legal line + irreversible majority')]
    for x,y,r,b,title,subtitle in boxes:
        d.rounded_rectangle((x,y,r,b),radius=18,fill='#edf0e6',outline='#98ac98',width=3)
        centered(d,((x+r)/2,y+50),title,face(37,True));centered(d,((x+r)/2,y+110),subtitle,face(26))
    d.line((870,640,1050,640),fill=INK,width=6);d.polygon([(1065,640),(1034,624),(1034,656)],fill=INK)
    d.line((1440,730,1440,795),fill=INK,width=6);d.polygon([(1440,810),(1424,779),(1456,779)],fill=INK)
    d.text((115,815),'SQLite saves completed analysis depths.',font=face(29),fill=INK)
    d.text((115,870),'JSON checkpoints save the finder frontier.',font=face(29),fill=INK)
    d.text((115,925),'Neural training is a separate research path.',font=face(29),fill=INK)
    im.save(ASSETS/'engine-flow.png')
architecture_figure()

PAPER_URL='https://www.researchgate.net/publication/239522973_Human_versus_Machine_Problem-Solving_Winning_Openings_in_Dakon'
MINISTRY_URL='https://kebudayaan.kemdikbud.go.id/ditwdb/congklak-betawi/'

def P(text,kind='p'):return {'kind':kind,'text':text}
def H(text):return P(text,'h2')
def T(headers,rows,widths=None):return {'kind':'table','headers':headers,'rows':rows,'widths':widths}
def I(name,caption,width=6.3):return {'kind':'image','file':name,'caption':caption,'width':width}
def C(text):return P(text,'code')

rules=[
 ['Own store','Leave seeds there and choose another nonempty own pit, subject to an empty-row pass.'],
 ['Previously occupied small pit on either row','Pick up every seed in that pit, including the last deposit, and continue sowing.'],
 ['Previously empty own pit with a nonempty opposite','Capture the landing seed and the opposite seeds into the own store. The turn ends.'],
 ['Previously empty own pit with an empty opposite','Leave the landing seed in its pit. The turn ends.'],
 ['Previously empty opposing pit','Leave the landing seed in that pit. The turn ends.'],
]
trace_rows=[[r['choice'],f'A{r["display_pit"]}',r['store'],r['store_gain'],r['relays'],r['captured']] for r in TRACE]
summary_rows=[]
for n in (1,2,3,7,8,18,32,64):
    r=ROWS[n-1];summary_rows.append([n,r['total'],r['required'],r['choices'] or '-',r['starter_store'] or '-',r['simulated_actions']])

report=[
 {'title':'Congklak engine and verified opening wins','blocks':[
 P('Technical report | Congklak engine project | 17 October 2026','small'),
 H('Abstract'),
 P('We built a Congklak analysis engine and a separate opening finder for an explicitly defined sequential, single-round game. From the seven-pit opening, the finder produces a legal 13-choice sequence that banks 54 of 98 seeds before the opponent chooses. A separate literal simulator verifies every choice, relay, capture and store count. The banked majority guarantees the starting player the round under the declared rules.'),
 P('For this report, we ran a bounded serial sweep over all board sizes n = 1 through 64, with n seeds per pit. It found and independently verified a majority-winning witness for every n = 2 through 64. The n = 1 search exhausted without a first-turn witness; its only initial choice was replayed separately. These are 63 finite, constructive results. They do not establish a formula or an existence theorem for unbounded n.'),
 H('What succeeded'),
 P('The project meets the opening-solver objective: find one legal first-turn line that secures a win, publish its complete replay, and allow the search to survive interruption. The proof depends on exact rules, integer seed counts, independent replay and permanent stores. It does not depend on a neural prediction or on winning a sample of games against another engine.'),
 T(['Result','Recorded evidence'],[['Seven-pit witness','13 choices; 54 banked; opponent maximum 44'],['Full finite sweep','63 verified witnesses at n = 2 through 64'],['Search work','34,795 simulated choices across all 64 sizes'],['Recorded sweep wall time','9.218 seconds on the reported workstation'],['Publication scope','Technical result and reproducible implementation; novelty is not established']],[1.6,4.7]),
 H('How to read the result'),
 P('A choice is the initial selection of one pit. A relay is a compulsory continuation within that choice. A turn may contain many choices when sowing repeatedly ends in the player\'s own store. The 13-choice line is not 13 alternating turns and is not claimed to be shortest. [E1-E4]'),
 ]},
 {'title':'Exact rules of the reported game','blocks':[
 P('The seven-pit profile is identified as id-betawi-relay7-v0. It is a declared engineering contract informed by the selected sequential Betawi description [R1]. Several regional and household details remain conventions of this implementation. The results apply to this complete contract rather than to every game called Congklak or Dakon.'),
 H('Board and legal choices'),
 P('Players A and B each own seven small pits and one store. Each small pit starts with seven seeds, both stores start empty, and the total is 98. The chosen starter moves alone. A legal initial choice selects a nonempty small pit on the player\'s own row. Empty pits, opposing pits and stores cannot be selected.'),
 P('Sow one seed at a time into successive locations, starting after the selected pit. Include the own store, skip the opposing store, and allow a full circuit to revisit the emptied source. Relays can occur on either row. A pit is "empty" according to its count immediately before the final deposit.'),
 T(['Final deposit','Resolution'],rules,[1.8,4.5]),
 H('Passing ending and score'),
 P('After a complete action, a player due to choose with an empty row must pass if the other row is nonempty. This applies after a store bonus as well as after an ordinary handoff. An empty row can later receive seeds again. Both rows must be empty for the round to end; there is no one-row sweep. Stored seeds are permanent. The greater final store wins, and 49-49 is a draw.'),
 P('The profile has no simultaneous opening, refills, closed pits, hidden counts or capture-before-a-lap restriction. An own-empty capture requires a nonempty opposite and includes the landing seed. Those choices are stated explicitly so the result remains reproducible. [E1]'),
 P('Display coordinates put A\'s store on the screen\'s left and B\'s on its right. The top row reads B7 to B1; the bottom row reads A1 to A7. Opposites are A1/B7 through A7/B1. Mirroring changes labels and layout, not the internal state.','small'),
 ]},
 {'title':'Engine architecture and search','blocks':[
 I('engine-flow.png','Figure 1. The general position analyzer and the opening finder have different search objectives. The verifier replays the finder\'s output independently.',6.3),
 H('Position analyzer'),
 P('The native C++20 simulator applies complete legal actions and reports relays, seed drops, captures, bonus choices and passes. A Python coordinator validates stable positions, manages a single worker, supplies explanations, and stores completed analysis depths in SQLite. The product uses the CPU H_v1 evaluator with iterative deepening, alpha-beta search, a transposition table and principal variation search.'),
 P('A transposition table reuses search results when different move orders reach a relevant stored position. Principal variation search first examines the most promising continuation, then tests alternatives with narrower bounds and re-searches when needed. The search accounts for the same player choosing again after a bonus. Its finite-depth scores remain estimates; only a separately established exact fact is presented as a guarantee.'),
 H('Generalized opening finder'),
 P('The n = 1 through 64 results come from the project\'s separate standard-library Python finder, not from running a 64-pit board through the seven-pit browser analyzer or its neural network. This finder uses exact integer counts and distributes full sowing circuits arithmetically. It then explores legal pit choices by depth-first backtracking until one sequence reaches the chosen banked-seed threshold.'),
 P('A bounded FIFO cache retains only fully searched failed states. Cache eviction may cost work but does not convert an unfinished branch into a failure. This search never needs to rank every possible opening by playing strength. It stops after producing one independently verified witness. [E2, E3]'),
 ]},
 {'title':'The verified seven pit sequence','blocks':[
 P('In the preferred left-store display, the sequence is <b>A7 A6 A3 A6 A7 A3 A6 A6 A5 A4 A5 A7 A7</b>. Its original right-store labels are 1, 2, 5, 2, 1, 5, 2, 2, 3, 4, 3, 1, 1. For canonical zero-based index i, the left-store label is 7 - i.'),
 T(['Choice','Pit','A banked','Gain','Relays','Captured'],trace_rows,[.65,.7,1.1,.75,1.0,1.1]),
 P('The first twelve choices end in A\'s store and leave A with 49 seeds. Choice 13 contains three relays and 17 seed drops. It adds five seeds to A\'s store in total, including a capture of four seeds. That capture ends the turn with A holding 54. Player B has made no choice. Across the full line there are 86 relays and 637 seed drops. [E4]'),
 H('Why this is a guaranteed win'),
 P('The board contains exactly 98 seeds and nothing removes seeds from a store. After A banks 54, all other locations together contain only 44. Even assigning every one of those remaining seeds to B leaves B behind A. The round may continue, but its winner is already secured under the specified scoring rule. No assumption about B\'s skill or replies is needed.'),
 ]},
 {'title':'Winning position and banked seeds','blocks':[
 P('The certificate ends at a stable position after choice 13. A has 54 seeds in its store and B has zero. The remaining pits hold 29 on A\'s row and 15 on B\'s row. The total remains 98. The 44-seed bound for B deliberately assigns every remaining pit seed to B, including those currently on A\'s row.'),
 I('board-13.png','Figure 2. The verified position after the final capture. The turn has ended, but the store majority is already irreversible.',6.1),
 I('banked-seeds.png','Figure 3. A\'s banked count after each complete choice. Choice 12 leaves 49, and choice 13 crosses the 50-seed target and finishes at 54.',6.1),
 P('The supplied video frames preserve these exact counts. They show stable positions after complete pit choices, including every relay in the selected choice. They do not animate individual seed drops. [E4]','small'),
 ]},
 {'title':'Independent verification and recovery','blocks':[
 H('What the checker verifies'),
 P('The searcher uses bulk arithmetic for complete circuits. The verifier uses a separate implementation that deposits one seed at a time and never imports the searcher. For each certificate it reconstructs the initial board and checks the selected pit, all successor counts, relays, captures, next controller, passes and terminal flag. It requires genuine own-store continuations before the last choice and checks that the opponent had no earlier choice.'),
 P('The verifier also checks total seeds, the goal, the final bank, the opponent\'s maximum possible total, the rule record and the certificate checksum. The checksum detects changes; the literal replay and majority argument establish the result. Shared rule declarations do not by themselves prove that the rules match every real-world tradition.'),
 H('Verification for this report'),
 P('All 63 successful sweep certificates were checked again in 63 fresh verifier processes. Each process ran congklak.dakon_opening_verify without using the optimized transition implementation. The recorded certificate hashes, source identities and outputs are retained in the evidence bundle. At n = 1 the sole initial action was checked separately: A banks one seed, has an empty row, and passes to B before reaching the target of two. [E3, E5]'),
 H('How interrupted work survives'),
 P('The analyzer commits complete search depths to SQLite. An unfinished depth is recomputed on recovery. The finder saves its depth-first frontier, explored-choice indices, bounded failed-state cache and counters to alternating checksummed JSON checkpoint slots. It writes a temporary record, flushes it and atomically replaces a slot. Recovery can use the older valid slot when the newest is corrupt.'),
 P('A discovered line is saved as pending verification before it is certified. Cancellation, a time limit or an action limit returns unknown, rather than a false win or a claim of impossibility. A completed run can reopen without changing its certificate. The retained CLI tests include four actual process kills; browser-integration tests include three further server kills at checkpoint and certificate-publication boundaries. [E6, E7]'),
 P('These checks establish independence between the two simulation implementations and test process-interruption recovery. They are not an external laboratory replication or a test involving physical power removal.','small'),
 ]},
 {'title':'Finding witnesses from n 1 through n 64','blocks':[
 P('For each n, initialize n seeds in each of the 2n small pits. Total seeds are 2n² and the majority threshold is n² + 1. Only the starting player chooses along a successful first-turn witness. Intermediate choices must finish in the own store, without a pass or handoff; a final capture is allowed if it reaches the target.'),
 C('SEARCH(position, line)\n  for each filled own pit in canonical order\n    successor = exact complete sowing action\n    if own_store(successor) >= n*n + 1\n      save line; verify; publish and return success\n    else if genuine own-store bonus and no pass\n      return on child success; otherwise backtrack\n  cache a failed state only after all choices fail'),
 P('The report campaign used starter A, canonical ascending pit order, a 10,000-entry cache, a 100,000-choice action cap and a three-second initial time budget per size. Checkpoints occurred every 256 tested choices or roughly one second. Every size completed within that initial invocation; no time-limited or pending case remained. [E3]'),
 T(['n','Total','Target','Choices','Banked','Tested choices'],summary_rows,[.45,.75,.8,.85,.85,1.45]),
 P('The serial sweep recorded 34,795 simulated choices in 9.218 seconds. The largest tested-choice count for one size was 1,238; the longest size invocation was 0.469 seconds. The host was an Intel Core i9-13900K, 24 cores and 32 logical processors, about 32 GB RAM, Windows 11 and Python 3.12.14. One CPU worker and no GPU were used. This single timing includes checkpoints and in-search verification, but excludes process startup and the later standalone audit. It is not a repeated benchmark or an energy measurement.'),
 P('n = 2 through 64 each has a verified witness. n = 1 has no uninterrupted first-turn majority witness in this search; this is not a claim that its whole game is a loss. The finite sweep does not establish a theorem for every larger n. No run minimizes sequence length. The complete rows appear on the next page and in results.csv.'),
 ]},
 {'title':'Complete finite sweep results','blocks':[
 P('All rows use the majority objective, starter A and the same source versions. Banked is the verified store count at the target; tested is the number of fully simulated search choices. The n = 1 row has no winning certificate.','small'),
 {'kind':'paired-results'},
 P('Verified witness rows are n = 2 through 64 inclusive. The 63 certificates together establish this finite range for the declared rules. They do not establish shortestness, exhaustive counts of winning lines, stronger refill goals, or a result for n greater than 64. [E3, E5]','small'),
 ]},
 {'title':'Comparison with earlier Dakon work','blocks':[
 T(['Detail','Dakon paper [R2]','This project'],[
 ['Match format','Repeated refill rounds; unfillable pits close','One round; permanent stores'],
 ['Cannot move','Round ends; remaining seeds swept','Mandatory pass; both rows must empty'],
 ['Opening objective','Prevent the opponent filling one pit next round','Bank a strict majority in this round'],
 ['Store landing and relay','Store permits another choice; occupied pits relay','Same broad mechanisms'],
 ['Own empty landing','Wording captures landing and opposite seeds','Capture only when opposite is nonempty'],
 ['Search','Backtracking; first winners and special objectives','First verified witness; recoverable frontier'],
 ],[1.1,2.5,2.7]),
 P('Donkers, de Voogt and Uiterwijk (2000) report winning openings at n = 2 and 4 through 18, with n = 3 failing their stronger objective. Their 50-million figure concerns enumerating sequences; they also report finding individual winners quickly. [R2]'),
 P('Our n = 3 majority witness banks 10 of 18 seeds. It answers a different threshold question and does not contradict that historical result. Likewise, a 13-choice majority witness cannot be ranked against a shortest line for the stronger refill objective. The empty-opposite wording is a possible implementation mismatch; the published prose alone does not establish the authors\' exact code behavior.'),
 P('The appropriate contribution is a reproducible implementation of a declared variant, a certified seven-pit line, a finite n = 2 through 64 witness set, and recoverable execution with an independent checker. We make no priority claim for discovering first-turn wins, no shortest-line claim, and no measured speedup over the historical program.'),
 H('Neural training as project background'),
 P('The broader project investigated supervised policy/value training, deeper teacher labels, relay-aware features, PUCT and recoverable self-play. Those experiments motivated data provenance, reservation and evaluation work. They are background to this report: the demonstrated opening proof uses deterministic exact simulation and backtracking, and loads no learned model. H_v1 remains the product default; this report does not require a claim of superior neural play. [E8]'),
 H('Success criterion'),
 P('The supported success statement is that the project finds and verifies a first-turn strategy securing the declared starting position. A general-purpose analyzer remains useful when users leave the certified line or enter other positions. Its ordinary finite-depth estimates are distinct from the opening certificate.'),
 ]},
 {'title':'Reproduction and evidence','blocks':[
 P('Run the commands from the repository root with Python 3.12. The sweep wrapper creates or resumes only its dedicated report campaign below data/dakon-opening-v1/report-20261017. Existing complete runs reopen rather than being replaced. For a new timing study, use a fresh campaign directory; reopening completed runs is not a fresh search timing.'),
 C('python -B tools/report-opening-sweep-v1.py\npython -B tools/report-opening-audit-v1.py\n\n# Independently check the seven-pit certificate\n$cert = "data/dakon-opening-v1/report-20261017/" +\n        "n07-majority-A/certificate.json"\npython -B -m congklak.dakon_opening_verify $cert'),
 P('For a completely new report sweep, copy the wrapper and change RUNS to another dedicated subdirectory under data/dakon-opening-v1. Keep the pinned searcher and verifier sources unchanged when resuming. Each certificate includes the complete rule record, source hashes, zero-based actions, every stable successor and a proof summary.'),
 H('Project evidence'),
 P('[E1] rules/id-betawi-relay7-v0.json and docs/CONGKLAK_RULES_REVIEW.md. Full executable rules and human-readable conventions.','small'),
 P('[E2] congklak/config.py, congklak/dakon_opening.py and congklak/dakon_opening_verify.py. Analyzer identity, exact first-witness search and literal checker.','small'),
 P('[E3] build/report-openings-v1/sweep.json and results.csv. All 64 invocations, work counters, budgets, environment and certificate file hashes.','small'),
 P('[E4] build/report-openings-v1/seven-pit-trace.json and data/dakon-opening-v1/report-20261017/n07-majority-A/certificate.json. Complete winning trace.','small'),
 P('[E5] build/report-openings-v1/independent-verification.json. All 63 standalone verifier results plus n = 1 literal replay.','small'),
 P('[E6] build/dakon-opening-v1/acceptance.json. 18 finder tests, four real process kills and 56 native/reference opening comparisons.','small'),
 P('[E7] build/product-finder-v1/acceptance.json and recovery.json. 48 Python tests and 68 frontend checks in the integration acceptance; three real finder-server recovery cases. The recorded live activation was still pending a manual restart at the report cutoff.','small'),
 P('[E8] README.md and build/benchmark-loss-review-v1/product-readiness-final.json. Background on the analyzer and experimental neural paths. Historical experiment findings are not opening proof inputs.','small'),
 H('Published and cultural references'),
 P(f'[R1] Ministry of Education and Culture. Congklak Betawi. <link href="{MINISTRY_URL}">Source page</link>. The project supplies an explicit complete rules contract alongside this cultural reference.','small'),
 P(f'[R2] Donkers J, de Voogt A, Uiterwijk J. Human versus Machine Problem-Solving Winning Openings in Dakon. Board Games Studies 3 (2000), 79-88. <link href="{PAPER_URL}">Author-uploaded full text</link>.','small'),
 P('All finite-sweep certificates and source identities are included in the accompanying evidence archive. Neural weights, training corpora and private saved games are not needed to reproduce the reported opening results.','small'),
 ]},
]

# Timings are production estimates, not synchronized subtitles.
chapters=[
 ('00 00 to 00 45','A win before the other player chooses',
 'Open on board-13.png. Keep the caption "Seven pits sequential single round" visible. Show 54 secured against a maximum of 44, then cut to the empty stores on board-00.png.',
 "We started this project wanting a computer to analyze Congklak. We ended up with something very concrete: a sequence that secures a win before the other player even chooses a pit. On this board there are 98 seeds. Our starting player can bank 54 of them during the opening turn. That leaves only 44 seeds everywhere else. The opponent can never catch up under these rules. We have the actual sequence, and a second program that checks every seed movement. Let me show you the rules, how the search found it, and exactly what this result means."),
 ('00 45 to 02 00','The exact game we tested',
 'Use board-00.png. Point to A7, A1, both stores and the two rows. Add short overlays for relay bonus capture and pass. Keep the rules caption throughout the demonstration.',
 "Congklak has regional and household variations, so the rules matter. Our board has seven small pits on each side and seven seeds in each pit. The stores begin empty. We use a sequential opening: one player starts, and the other waits. From your seat the store is on your left. You choose a filled pit on your own row and sow one seed at a time, including your own store and skipping the opponent's store. If your final seed lands in an occupied small pit, pick up that pit and continue. That is a relay inside the same choice. If the final seed lands in your store, choose again. If it lands in an empty own pit opposite some seeds, capture the landing seed and those opposite seeds, then end the turn. With an empty opposite, leave the landing seed where it is. An empty row causes a pass. We play a single round until both rows are empty, with no refill and no early sweep. All the seeds you bank stay banked."),
 ('02 00 to 02 50','What our engine actually does',
 'Show engine-flow.png. Emphasize the analyzer branch first, then the opening finder and separate verifier. Neural training receives one background sentence only.',
 "The project has two useful search tools. The position analyzer uses a C++ simulator and a CPU search that looks ahead through legal choices. It reuses positions in a transposition table and saves completed depths so interrupted analysis can resume. Its normal scores are estimates. For this opening task, we use a simpler objective: find one legal line that banks a majority before the opponent chooses. That tool explores choices by backtracking and stops as soon as an independently checked witness succeeds. We also experimented with neural training and self-play, but those experiments are not needed for the result you are about to see. The opening certificate comes from exact seed counts."),
 ('02 50 to 04 25','Follow the thirteen choices',
 'Sequence board-00.png through board-13.png. Hold longer after choices 1 2 7 12 and 13. Show the complete pit sequence and use banked-seeds.png for the rising store count.',
 "Here is the line using our left-store labels: A7, A6, A3, A6, A7, A3, A6, A6, A5, A4, A5, A7, A7. These are choices, not alternating turns. The first choice moves seven seeds straight into the six pits in front of A7 and then A's store. A has banked one seed and chooses again. Some later choices contain many relays, so one selected pit can move a lot of seeds before the hand becomes empty. After choice two, A has eight in the store. After choice seven, the count is 30. The first twelve choices all end in the store, giving A another legal choice each time. After the twelfth, A has 49, just short of the 50 needed to secure a majority. The thirteenth choice has three relays and ends with a capture of four seeds. The complete choice adds five to the store and leaves A with 54. The capture ends the turn. B has not chosen a pit at any point in this sequence."),
 ('04 25 to 05 25','How we know the result is correct',
 'Hold board-13.png. Write 98 minus 54 equals 44 on screen. Show the independent-verification.json passed record and a short terminal verification command without unrelated logs.',
 "The guarantee is arithmetic, once the sequence has been checked. There are 98 seeds in total. A already holds 54 permanently. Even if B eventually collected all the other 44 seeds, B would still finish behind. We do not have to guess B's replies to establish that. But the seed movements must be correct. The searcher speeds up long circuits with arithmetic. Our second simulator instead drops seeds one at a time. It rebuilds the starting board and checks every chosen pit, relay, capture, store count and player handoff. It rejects a line if the opponent gets a choice before the target is reached. That independent replay is what lets us present a checked winning certificate rather than a high evaluation score."),
 ('05 25 to 06 35','Checking the finite range up to sixty four',
 'Show coverage.png, then the results CSV or the report table. Highlight n1 n3 n7 and n64. Keep "Finite checked range" on screen.',
 "We also asked the same question on boards with n pits per side and n seeds in each pit. That gives two times n squared seeds in total, so n squared plus one is enough for a majority. For this report we ran every size from one through 64. The finder produced a winning certificate for every size from two through 64, and all 63 certificates passed fresh standalone verifier processes. At three pits, our majority witness banks 10 of 18 seeds. At 64 pits, it banks 4,103 of 8,192 in 265 choices. The one-pit search exhausted without a first-turn witness; its only choice was checked separately. This is a complete finite table, not a theorem covering every possible board size. The recorded sweep took about nine seconds on our i9-13900K workstation, including checkpoints and in-search verification. Startup and the later standalone audit are excluded. That is one recorded run, not a comparison with another engine."),
 ('06 35 to 07 15','How this relates to the Dakon paper',
 'Show the paper title and citation as plain text, then the report comparison table. Do not show a 13 versus 25 fastest or shortest comparison.',
 "First-turn winning openings were already studied by Donkers, de Voogt and Uiterwijk in 2000. Their Dakon uses refill rounds and a stronger opening target. Our single-round majority question differs. So we credit that earlier work and report our exact rules, certificates and finite results without claiming to be the first to discover the idea."),
 ('07 15 to 08 15','The practical result',
 'Return to the exact seven-pit sequence, then show a paused saved run and replay controls if the local finder backend is active. Otherwise use the included stable images and recorded CLI results. End on the certificate and evidence archive.',
 "For this project, success means finding a winning strategy we can reproduce and explain. We do not need a neural network to outperform another player when a legal opening line already secures a majority. The finder stops at one verified witness, and it saves its frontier so an interrupted search can continue. The analyzer is still useful when a person chooses a different opening or enters another position. Those ordinary evaluations are a separate capability. Our demonstrated result is specific and checkable: under these seven-pit sequential single-round rules, the starter can follow a 13-choice sequence and bank 54 seeds before the opponent chooses. It does not mean every first-player move wins, that the line is shortest, or that every Congklak tradition has the same outcome. The report, rules, code and certificates make the claim inspectable. If you try the sequence, use the same labels and rules so you are checking the same game."),
]

video=[{'title':'YouTube script for the Congklak opening result','blocks':[
 P('Narration and recording plan | English | Approximately eight to ten minutes','small'),
 P('The story follows the verified seven-pit result first, then explains the engine, independent verification and finite board-size sweep. The time ranges below are editing estimates. Record the voiceover naturally and retime cuts after recording.'),
 H('Suggested title'),P('We Found a Guaranteed Winning Opening in Congklak'),
 H('Thumbnail wording'),P('54 SEEDS BEFORE A REPLY'),
 H('Persistent rule caption'),P('7 pits per side | 7 seeds per pit | Sequential | Single round'),
 H('Essential recording facts'),
 P('The certified sequence is A7 A6 A3 A6 A7 A3 A6 A6 A5 A4 A5 A7 A7 in the left-store view. A banks 54 of 98. The first twelve choices end in the store; the last ends in capture. A turn contains several choices, and one choice can contain several relays.'),
 P('Use the finite-range wording "verified for n = 2 through 64". At n = 1, say "no first-turn witness in the completed restricted search". Refer to a winning strategy for the starter, rather than saying that any first-player move wins.'),
 H('Visual assets'),
 P('The video-assets directory contains board-00.png through board-13.png, banked-seeds.png, coverage.png and engine-flow.png. All are original 1920 by 1080 images derived from the certificate. Board frames show complete choices, not individual seed-drop animations. No third-party paper images are needed.'),
 ]}]
for i in range(0,len(chapters),2):
    blocks=[]
    for timing,title,shot,narration in chapters[i:i+2]:
        clock=re.sub(r'(\d{2}) (\d{2})',r'\1:\2',timing)
        blocks.extend([H(title),P('Approximate time '+clock+'. Visual plan: '+shot,'small'),P(narration)])
    video.append({'title':f'Narration part {i//2+1}','blocks':blocks})
video.append({'title':'Recording checklist and video description','blocks':[
 H('Recording checklist'),
 P('Check the pit labels before recording: A7 is the far-right bottom pit and A1 is next to A\'s left store. Keep the rule caption visible during the line. Allow relays to finish before advancing to the next numbered choice. Highlight A\'s store at 49 after choice 12, then show 54 and the capture handoff after choice 13.'),
 P('Use certificate evidence for the proof demonstration, not an ordinary H_v1 recommendation or an uncalibrated win probability. If showing the interactive finder, check that /api/finder works in the local app first; the retained activation record still required a manual restart. No restart or saved-game change was performed while preparing this report.'),
 P('Do not describe nine seconds as an energy result or a speedup over historical hardware. Do not present support for 64 sizes as an unbounded theorem. Avoid calling the 13-choice line shortest. Keep neural training to the background sentence in the narration.'),
 H('Suggested description'),
 P('We built a Congklak analyzer and an exact opening finder. Under our declared seven-pit sequential single-round rules, a verified 13-choice line banks 54 of 98 seeds before the opponent chooses. We also checked every n from 1 to 64 with n seeds per pit: majority-winning certificates were found and independently verified at n = 2 through 64. The report includes the exact rules, trace, full finite results and reproduction commands.'),
 P('Evidence: include the report PDF and evidence archive from this package in your chosen public repository or download location before publishing. Those local files have not been uploaded or assigned a public link.'),
 H('References for the description'),
 P(f'Ministry reference: <link href="{MINISTRY_URL}">Congklak Betawi</link>. Earlier research: <link href="{PAPER_URL}">Donkers et al 2000</link>. The complete project rules and machine-checkable evidence accompany this script.','small'),
 H('Optional closing prompt'),
 P('Have you played a version with different capture or opening rules? Describe the exact difference so it can be tested as a separate variant.'),
 ]})

def table_flow(block,compact=False):
    data=[[Paragraph(html.escape(str(c)),styles['th']) for c in block['headers']]]
    centered_cell=ParagraphStyle('centered-cell',parent=styles['cell'],alignment=TA_CENTER)
    data += [[Paragraph(html.escape(str(c)),centered_cell if isinstance(c,(int,float)) or c=='-' else styles['cell']) for c in row] for row in block['rows']]
    widths=[x*72 for x in block['widths']] if block.get('widths') else None
    table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    settings=[('BACKGROUND',(0,0),(-1,0),NAVY),('GRID',(0,0),(-1,-1),.5,GRAY),
              ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),
              ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),
              ('BOTTOMPADDING',(0,0),(-1,-1),5)]
    for row in range(1,len(data)):
        if row%2==0:settings.append(('BACKGROUND',(0,row),(-1,row),colors.HexColor('#f2f4f5')))
    table.setStyle(TableStyle(settings))
    return table

def result_pair():
    tables=[]
    for start,end in [(0,32),(32,64)]:
        data=[['n','Choices','Banked','Target','Tested']]
        for r in ROWS[start:end]:data.append([r['n'],r['choices'] or '-',r['starter_store'] or '-',r['required'],r['simulated_actions']])
        table=Table(data,colWidths=[24,44,45,45,49],repeatRows=1)
        table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,0),'BodyBold'),('FONTNAME',(0,1),(-1,-1),'Body'),
            ('FONTSIZE',(0,0),(-1,-1),8.5),('LEADING',(0,0),(-1,-1),10.5),
            ('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('GRID',(0,0),(-1,-1),.5,GRAY),('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),
            ('BOTTOMPADDING',(0,0),(-1,-1),4),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f4f5')])]))
        tables.append(table)
    pair=Table([[tables[0],tables[1]]],colWidths=[226,226],hAlign='LEFT')
    pair.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),15)]))
    return pair

def footer(c,doc):
    c.setFont('Body',8);c.setFillColor(colors.HexColor('#666666'))
    c.drawString(53,29,'Congklak engine project   17 October 2026')
    c.drawRightString(A4[0]-53,29,str(doc.page))

def build_pdf(pages,name):
    story=[]
    for idx,page in enumerate(pages):
        if idx:story.append(PageBreak())
        title=page['title']
        if idx==0 and title=='Congklak engine and verified opening wins':
            title='Congklak engine<br/>and verified opening wins'
        story.append(Paragraph(title,styles['title' if idx==0 else 'h1']))
        for block in page['blocks']:
            kind=block['kind']
            if kind=='table':story.extend([table_flow(block),Spacer(1,12)])
            elif kind=='paired-results':story.extend([result_pair(),Spacer(1,12)])
            elif kind=='image':
                width=block['width']*72
                story.append(PDFImage(str(ASSETS/block['file']),width=width,height=width*1080/1920))
                story.append(Paragraph(block['caption'],styles['caption']))
            elif kind=='code':story.append(Preformatted(block['text'],styles['code']))
            else:story.append(Paragraph(block['text'],styles.get(kind,styles['p'])))
    SimpleDocTemplate(str(OUT/name),pagesize=A4,rightMargin=53,leftMargin=53,
                      topMargin=43,bottomMargin=45,title=pages[0]['title'],author='Congklak engine project').build(story,onFirstPage=footer,onLaterPages=footer)

def build_html(pages,name):
    content=[]
    for i,page in enumerate(pages):
        content.append(f'<section><h1>{html.escape(page["title"])}</h1>')
        for b in page['blocks']:
            kind=b['kind']
            if kind in ('table','paired-results'):
                rows=b.get('rows',[[r['n'],r['choices'] or '-',r['starter_store'] or '-',r['required'],r['simulated_actions']] for r in ROWS])
                headers=b.get('headers',['n','Choices','Banked','Target','Tested'])
                content.append('<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>')
                content.extend('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)
                content.append('</tbody></table>')
            elif kind=='image':content.append(f'<figure><img src="video-assets/{b["file"]}" alt="{html.escape(b["caption"])}"><figcaption>{b["caption"]}</figcaption></figure>')
            elif kind=='code':content.append('<pre>'+html.escape(b['text'])+'</pre>')
            else:
                tag='h2' if kind=='h2' else 'p'
                text=b['text'].replace('<link href=','<a href=').replace('</link>','</a>')
                content.append(f'<{tag}>{text}</{tag}>')
        content.append('</section>')
    css='body{max-width:880px;margin:45px auto;padding:0 28px;font:17px/1.65 Arial,sans-serif;color:#111}h1,h2{color:#000}h1{font:36px/1.2 Georgia,serif}h2{font-size:21px;margin-top:30px}section{margin-bottom:60px}table{width:100%;border-collapse:collapse;font-size:14px;margin:20px 0}td,th{border:1px solid #d9d9d9;padding:9px;vertical-align:middle}th{background:#263b47;color:white;text-align:left}tr:nth-child(even){background:#f2f4f5}img{width:100%;height:auto}figure{margin:22px 0}figcaption{font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}a{color:#254c71}@media print{section{break-before:page}section:first-child{break-before:auto}body{margin:0;font-size:11pt}}'
    (OUT/name).write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(pages[0]['title'])+'</title><style>'+css+'</style><body>'+''.join(content)+'</body></html>',encoding='utf-8')

build_pdf(report,'Congklak-engine-report.pdf')
build_pdf(video,'YouTube-script-and-recording-plan.pdf')
build_html(report,'Congklak-engine-report.html')
build_html(video,'YouTube-script-and-recording-plan.html')
(OUT/'editable-content.json').write_text(json.dumps(dict(report=report,video=video),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
shutil.copy2(EVIDENCE/'results.csv',OUT/'results-n1-to-n64.csv')
evidence_out=OUT/'evidence';evidence_out.mkdir(exist_ok=True)
for file in ('sweep.json','independent-verification.json','seven-pit-trace.json'):
    shutil.copy2(EVIDENCE/file,evidence_out/file)
for file in ('congklak/dakon_opening.py','congklak/dakon_opening_verify.py','rules/id-betawi-relay7-v0.json',
             'tools/report-opening-sweep-v1.py','tools/report-opening-audit-v1.py'):
    target=evidence_out/file;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/file,target)
for row in ROWS:
    run=ROOT/row['run_dir'];dest=evidence_out/'runs'/f'n{row["n"]:02d}';dest.mkdir(parents=True,exist_ok=True)
    for file in ('identity.json','status.json','checkpoint-0.json','checkpoint-1.json','certificate.json'):
        if (run/file).exists():shutil.copy2(run/file,dest/file)
manifest={'report_date':'2026-10-17','verified_n':list(range(2,65)),'no_first_turn_witness_n':[1],
          'not_an_unbounded_theorem':True,'seven_pit_choices':13,'seven_pit_banked':54,
          'independent_verifier_processes':63,'new_neural_training_steps':0,
          'source_sha256':SWEEP['source_sha256'],
          'figures':sorted(p.name for p in ASSETS.glob('*.png')),
          'render_qa':'pending',
          'files':{p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name not in ('manifest.json','Congklak-report-and-video-kit.zip')}}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'report':str(OUT/'Congklak-engine-report.pdf'),'script':str(OUT/'YouTube-script-and-recording-plan.pdf'),
                  'report_pages_planned':len(report),'script_pages_planned':len(video),'assets':len(manifest['figures'])}),flush=True)
