"""Own-process Windows performance-core affinity for reproducible experiments.

Topology: https://learn.microsoft.com/windows/win32/api/winnt/ns-winnt-system_cpu_set_information
Affinity: https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-setprocessaffinitymask
No power, priority, or other process settings are changed.
"""
from __future__ import annotations
import ctypes as C
from ctypes import wintypes as W
import os
import struct


def parse_cpu_sets(raw):
    rows=[]; offset=0
    while offset<len(raw):
        if len(raw)-offset<8: raise ValueError("Truncated CPU-set header")
        size,kind=struct.unpack_from("<II",raw,offset)
        if size<8 or offset+size>len(raw): raise ValueError("Invalid CPU-set record size")
        if kind==0:
            if size<32: raise ValueError("Truncated CPU-set record")
            cpu_id,group,logical,core,cache,numa,efficiency,flags=struct.unpack_from("<IHBBBBBB",raw,offset+8)
            rows.append(dict(id=cpu_id,group=group,logical=logical,core=core,cache=cache,numa=numa,
                             efficiency_class=efficiency,parked=bool(flags&1),allocated=bool(flags&2),allocated_to_process=bool(flags&4)))
        offset+=size
    if not rows: raise ValueError("No CPU sets available")
    return rows


def select_core(rows,allowed_mask):
    eligible=[r for r in rows if r["group"]==0 and allowed_mask&(1<<r["logical"])
              and not r["parked"] and (not r["allocated"] or r["allocated_to_process"])]
    if not eligible: raise ValueError("No permitted group-zero CPU set")
    highest=max(r["efficiency_class"] for r in eligible)
    performance=[r for r in eligible if r["efficiency_class"]==highest]
    # Avoid the first physical core when another is available; no hard-coded
    # P/E mapping or assumption about logical processor numbering is used.
    cores=sorted({r["core"] for r in performance})
    selected_core=cores[1] if len(cores)>1 else cores[0]
    return min((r for r in performance if r["core"]==selected_core),key=lambda r:r["logical"])


class Processor(C.Structure):
    _fields_=[("group",W.WORD),("number",C.c_ubyte),("reserved",C.c_ubyte)]


class PerformanceCore:
    """Pin only the calling process; restore its exact previous mask on exit."""
    def __enter__(self):
        if os.name!="nt": raise RuntimeError("This measured protocol requires Windows CPU-set topology")
        self.kernel=C.WinDLL("kernel32",use_last_error=True)
        k=self.kernel
        k.GetCurrentProcess.restype=W.HANDLE
        k.GetProcessAffinityMask.argtypes=[W.HANDLE,C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)]
        k.GetProcessAffinityMask.restype=W.BOOL
        k.SetProcessAffinityMask.argtypes=[W.HANDLE,C.c_size_t];k.SetProcessAffinityMask.restype=W.BOOL
        k.GetSystemCpuSetInformation.argtypes=[C.c_void_p,W.ULONG,C.POINTER(W.ULONG),W.HANDLE,W.ULONG]
        k.GetSystemCpuSetInformation.restype=W.BOOL
        k.GetCurrentProcessorNumberEx.argtypes=[C.POINTER(Processor)];k.GetCurrentProcessorNumberEx.restype=None
        self.handle=k.GetCurrentProcess();old=C.c_size_t();system=C.c_size_t()
        if not k.GetProcessAffinityMask(self.handle,C.byref(old),C.byref(system)): raise C.WinError(C.get_last_error())
        self.old_mask=old.value
        length=W.ULONG()
        k.GetSystemCpuSetInformation(None,0,C.byref(length),self.handle,0)
        if not length.value: raise C.WinError(C.get_last_error())
        buffer=C.create_string_buffer(length.value)
        if not k.GetSystemCpuSetInformation(buffer,len(buffer),C.byref(length),self.handle,0): raise C.WinError(C.get_last_error())
        self.rows=parse_cpu_sets(buffer.raw[:length.value])
        if any(r["group"]!=0 for r in self.rows): raise RuntimeError("Multi-group topology needs a separately declared protocol")
        selected=select_core(self.rows,self.old_mask);self.mask=1<<selected["logical"]
        if not k.SetProcessAffinityMask(self.handle,self.mask): raise C.WinError(C.get_last_error())
        self.metadata=dict(protocol="own-process/single-os-reported-performance-core/v1",selected=selected,
                           mask=self.mask,original_mask=self.old_mask,
                           efficiency_classes=sorted({r["efficiency_class"] for r in self.rows}),
                           logical_processors=len(self.rows),power_settings_changed=False,priority_changed=False)
        try: self.verify()
        except Exception as error:
            if not k.SetProcessAffinityMask(self.handle,self.old_mask):
                raise C.WinError(C.get_last_error()) from error
            raise
        return self

    def verify(self):
        actual=C.c_size_t();system=C.c_size_t()
        if not self.kernel.GetProcessAffinityMask(self.handle,C.byref(actual),C.byref(system)): raise C.WinError(C.get_last_error())
        p=Processor();self.kernel.GetCurrentProcessorNumberEx(C.byref(p))
        if actual.value!=self.mask or p.group!=0 or p.number!=self.metadata["selected"]["logical"]:
            raise RuntimeError("Benchmark process/thread left its declared CPU")
        return dict(group=int(p.group),logical=int(p.number),mask=actual.value)

    def __exit__(self,*_):
        if not self.kernel.SetProcessAffinityMask(self.handle,self.old_mask): raise C.WinError(C.get_last_error())
