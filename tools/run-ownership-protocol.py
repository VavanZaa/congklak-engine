"""Build or verify the recoverable prospective protocol definition."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak.ownership_protocol_flow import main
if __name__=='__main__':main()
