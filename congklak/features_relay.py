"""Actor-relative count and exact one-choice relay features, without target access.

127 inputs = the frozen 43 counts/discontinuities plus 14 hypothetical choices
in own-then-opponent pit order, six values each: acting store gain/98,
drops/98, relays/98, capture, retained normalized control, terminal.
Empty hypothetical pits are zero; a side is never implicitly passed.
Stores cannot affect the sowing path, termination, or store gain. Thus the
features depend only on the 14 pits and normalized actor; D is added exactly
outside the network. Drops/relays are deliberately not clipped.
"""
from functools import lru_cache
from . import config,features_v3
from .native import Native,NativeError,flatten

ENCODER43=features_v3.ENCODER43
ENCODER127="current-controller-count43-own-opponent-choice14-storegain-drops-relays-capture-retain-terminal84/store-offset-exact-v5"
FEATURE_POLICY="normalized-controller/hypothetical-own7-opponent7/exact-relay6-unclipped-v1"

@lru_cache(maxsize=1)
def _native():
    return Native()

def encoder(input_count):
    if type(input_count) is not int: raise ValueError("Input count must be integer")
    if input_count==43:return ENCODER43
    if input_count==127:return ENCODER127
    raise ValueError("Unknown relay feature count")

def encode(position,input_count=127,native=None):
    encoder(input_count)
    p=config.position(position)
    x,d,r=features_v3.encode(p,43)
    if input_count==43:return x,d,r
    n=native or _native()
    actor=p["controller"]
    if r and not sum(p["pits"][actor]):actor=1-actor
    for side in (actor,1-actor):
        for pit,count in enumerate(p["pits"][side]):
            if count==0:
                x.extend([0.0]*6)
                continue
            t=n.apply(flatten(p),actor=side,action=pit,total=98,trace=False)
            if not t.get("ok"):raise NativeError(str(t))
            x.extend([t["store_delta"][side]/98,t["drops"]/98,t["relays"]/98,
                      float(t["reason"]=="own_capture"),
                      float(not t["terminal"] and t["position"]["controller"]==side),
                      float(t["terminal"])])
    return x,d,r
