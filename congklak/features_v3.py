"""Cheap actor-relative count discontinuities; no labels or simulator calls."""
from . import config

ENCODER15="current-controller-pit14-and-active15/store-offset-exact-v1"
ENCODER43="current-controller-pit14-active15-occupied14-singleton14/store-offset-exact-v3"

def encode(position,input_count=43):
    if input_count not in (15,43): raise ValueError("Unknown feature count")
    p=config.position(position)
    active=sum(map(sum,p["pits"])); actor=p["controller"]
    if active and not sum(p["pits"][actor]): actor=1-actor
    counts=p["pits"][actor]+p["pits"][1-actor]
    x=[count/98 for count in counts]+[active/98]
    if input_count==43:
        x.extend(float(count>0) for count in counts)
        x.extend(float(count==1) for count in counts)
    return x,p["stores"][0]-p["stores"][1],active*(1 if actor==0 else -1)

def encoder(input_count):
    if input_count==15: return ENCODER15
    if input_count==43: return ENCODER43
    raise ValueError("Unknown feature count")
