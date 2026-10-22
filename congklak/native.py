"""ctypes boundary to the independently tested C++20 core; no transition logic here."""
from __future__ import annotations
import ctypes as C
import hashlib
import json
from pathlib import Path
from threading import RLock

ROOT = Path(__file__).resolve().parent.parent

def flatten(position: dict) -> list[int]:
    a, b = position["pits"]
    return list(a) + [position["stores"][0]] + list(b) + [position["stores"][1]]

class NativeError(RuntimeError):
    pass

class Control:
    def __init__(self, native, max_depth=16, time_ms=1000, node_budget=1000000, memory_mb=512,tt_mb=None,use_pvs=True):
        self.native = native
        self._lock = RLock()
        self._running = False
        self.handle = native.lib.cc_create_control(max_depth, node_budget, time_ms, memory_mb*1024*1024)
        if not self.handle:
            raise NativeError("invalid native search limits")
        tt_mb=max(0,min(16,memory_mb-16)) if tt_mb is None else tt_mb
        if type(tt_mb) is not int or not 0<=tt_mb<=1024 or type(use_pvs) is not bool:
            self.close()
            raise NativeError("TT size must be integer and PVS must be boolean")
        answer=native._decode(native.lib.cc_set_search_options(self.handle,tt_mb,int(use_pvs)))
        if not answer.get("ok"):
            self.close()
            raise NativeError(answer["error"]["message"])
        self.search_options={"tt_mb":tt_mb,"use_pvs":use_pvs}

    def cancel(self):
        with self._lock:
            if self.handle:
                self.native.lib.cc_cancel(self.handle)

    def close(self):
        with self._lock:
            if self._running:
                raise NativeError("cannot destroy a running control")
            if self.handle:
                self.native.lib.cc_destroy_control(self.handle)
                self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

class Native:
    def __init__(self, path=None):
        path = Path(path or ROOT / "build" / "congklak_native.dll")
        if not path.is_file():
            raise NativeError("native library missing; run tools/build.ps1 first")
        self.build_id = hashlib.sha256(path.read_bytes()).hexdigest()
        self.lib = C.CDLL(str(path))
        self.callback_type = C.CFUNCTYPE(C.c_int, C.c_char_p, C.c_void_p)
        ptr = C.POINTER(C.c_int)
        for name in ("cc_validate", "cc_normalize"):
            fn = getattr(self.lib, name)
            fn.argtypes = [ptr,C.c_int,C.c_int,C.c_int]
            fn.restype = C.c_char_p
        self.lib.cc_apply.argtypes = [ptr,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_void_p]
        self.lib.cc_apply.restype = C.c_char_p
        self.lib.cc_abi_version.argtypes=[]
        self.lib.cc_abi_version.restype=C.c_char_p
        self.abi = self.lib.cc_abi_version().decode("ascii")
        if self.abi != "congklak-native-v1":
            raise NativeError("incompatible native ABI")
        self.lib.cc_create_control.argtypes=[C.c_int,C.c_uint64,C.c_uint64,C.c_uint64]
        self.lib.cc_create_control.restype=C.c_void_p
        self.lib.cc_set_search_options.argtypes=[C.c_void_p,C.c_int,C.c_int]
        self.lib.cc_set_search_options.restype=C.c_char_p
        for name in ("cc_cancel", "cc_destroy_control"):
            getattr(self.lib,name).argtypes=[C.c_void_p]
            getattr(self.lib,name).restype=None
        self.lib.cc_analyze.argtypes=[ptr,C.c_int,C.c_int,C.c_int,C.c_void_p,C.c_int,self.callback_type,C.c_void_p]
        self.lib.cc_analyze.restype=C.c_char_p

    @staticmethod
    def _array(cells, pits, total):
        cells=list(cells)
        if len(cells) not in (2*pits+2,16) or len(cells)>16:
            raise NativeError("invalid ring array length")
        if any(type(x) is not int or not -(2**31)<=x<2**31 for x in cells):
            raise NativeError("counts must be 32-bit integers")
        if total is None:
            total=sum(cells)
        return (C.c_int*16)(*(cells+[0]*(16-len(cells)))), total

    @staticmethod
    def _decode(raw):
        if not raw:
            raise NativeError("null native response")
        return json.loads(raw.decode("utf-8"))

    def normalize(self, cells, pits=7, actor=0, total=None):
        arr, total=self._array(cells,pits,total)
        return self._decode(self.lib.cc_normalize(arr,pits,actor,total))

    def validate(self, cells, pits=7, actor=0, total=None):
        arr, total=self._array(cells,pits,total)
        return self._decode(self.lib.cc_validate(arr,pits,actor,total))

    def apply(self,cells,pits=7,actor=0,action=0,total=None,trace=True,control=None):
        arr,total=self._array(cells,pits,total)
        if control is None:
            return self._decode(self.lib.cc_apply(arr,pits,actor,total,action,int(trace),None))
        if control.native is not self:
            raise NativeError("control belongs to another native instance")
        with control._lock:
            if not control.handle or control._running:
                raise NativeError("control is closed or already running")
            control._running=True
        try:
            return self._decode(self.lib.cc_apply(arr,pits,actor,total,action,int(trace),control.handle))
        finally:
            with control._lock:
                control._running=False

    def analyze(self,cells,pits=7,actor=0,max_depth=16,time_ms=0,node_budget=0,
                start_depth=1,on_report=None,total=None,memory_mb=512,control=None,keep_history=True,tt_mb=None,use_pvs=True):
        arr,total=self._array(cells,pits,total)
        owned=control is None
        control=control or Control(self,max_depth,time_ms,node_budget,memory_mb,tt_mb,use_pvs)
        if control.native is not self:
            raise NativeError("control belongs to another native instance")
        reports=[]
        callback_error=[]
        def progress(raw,_):
            try:
                report=json.loads(raw.decode("utf-8"))
                if on_report is not None and on_report(report) is False:
                    return 0
                if not keep_history:
                    reports.clear()
                reports.append(report)
                return 1
            except Exception as exc:
                callback_error.append(str(exc))
                return 0
        callback=self.callback_type(progress)
        try:
            with control._lock:
                if not control.handle or control._running:
                    raise NativeError("control is closed or already running")
                control._running=True
            try:
                result=self._decode(self.lib.cc_analyze(arr,pits,actor,total,control.handle,start_depth,callback,None))
            finally:
                with control._lock:
                    control._running=False
            result["reports"]=reports
            if callback_error:
                result["callback_error"]=callback_error[0]
            return result
        finally:
            if owned:
                control.close()
