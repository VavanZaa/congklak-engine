"""Generate the requested pilot, deliberately kill once, then recover it."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/"data"/"pilot-10000"
LOG=ROOT/"build"/"phase2-generation.log"

def run(kill_after=None):
    child=subprocess.Popen([sys.executable,"-m","congklak.pilot","build","--path",str(DATA)],cwd=ROOT,
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding="utf-8")
    killed=False
    with LOG.open("a",encoding="utf-8") as log:
        for line in child.stdout:
            log.write(line); log.flush()
            event=json.loads(line)
            if kill_after and event.get("stage")=="label" and event.get("labels",0)>=kill_after:
                child.kill(); killed=True; break
            if event.get("stage") in ("complete",) or event.get("event")=="pilot_error": print(line.strip(),flush=True)
        child.wait(timeout=30)
        error=child.stderr.read()
        if error: log.write(error)
    child.stdout.close(); child.stderr.close()
    if not killed and child.returncode: raise RuntimeError("Generation failed; see "+str(LOG))
    return killed

def main():
    begin=time.monotonic(); LOG.parent.mkdir(parents=True,exist_ok=True)
    killed=run(1000)
    if killed: run()
    report={"actual_campaign_process_killed":killed,"kill_boundary":"after >=1000 complete durable labels", "elapsed_seconds":time.monotonic()-begin,
            "data_directory":str(DATA),"log":str(LOG)}
    (ROOT/"build"/"phase2-recovery.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report),flush=True)

if __name__=="__main__": main()
