"""Use the strict reader over the unchanged V2 protocol publication producer."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak.ownership_protocol_reader_v3 import main
if __name__=='__main__':main()
