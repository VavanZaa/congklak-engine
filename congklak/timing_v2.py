"""Explicit, stable CPU declarations for resumable Windows timing experiments.

https://learn.microsoft.com/windows/win32/api/winnt/ns-winnt-system_cpu_set_information
https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-setprocessaffinitymask
Only this process's affinity is changed; its exact previous mask is restored.
"""
from __future__ import annotations
import ctypes as C
from ctypes import wintypes as W
import os
import struct
import time

STRUCTURAL_FIELDS=("id","group","logical","core","cache","numa","efficiency_class")
PROTOCOL="own-process/declared-os-performance-core/v2"

def structural(row):return {key:row[key] for key in STRUCTURAL_FIELDS}

def declaration(value):
    if not isinstance(value,dict) or set(value)!=set(STRUCTURAL_FIELDS):raise ValueError("CPU declaration requires exactly the structural fields")
    limits={"id":2**32-1,"group":65535,"logical":63,"core":255,"cache":255,"numa":255,"efficiency_class":255}
    if any(type(value[key]) is not int or not 0<=value[key]<=limits[key] for key in STRUCTURAL_FIELDS):raise ValueError("Invalid structural CPU declaration")
    if value["group"]!=0:raise ValueError("This protocol supports an explicitly declared group-zero CPU")
    return dict(value)

def parse_cpu_sets(raw):
    rows=[];offset=0
    while offset<len(raw):
        if len(raw)-offset<8:raise ValueError("Truncated CPU-set header")
        size,kind=struct.unpack_from("<II",raw,offset)
        if size<8 or offset+size>len(raw):raise ValueError("Invalid CPU-set record size")
        if kind==0:
            if size<32:raise ValueError("Truncated CPU-set record")
            cpu_id,group,logical,core,cache,numa,efficiency,flags=struct.unpack_from("<IHBBBBBB",raw,offset+8)
            rows.append(dict(id=cpu_id,group=group,logical=logical,core=core,cache=cache,numa=numa,efficiency_class=efficiency,
                             parked=bool(flags&1),allocated=bool(flags&2),allocated_to_process=bool(flags&4)))
        offset+=size
    if not rows:raise ValueError("No CPU sets available")
    return rows

def select_core(rows,allowed_mask,preferred):
    """Locate one exact declaration; transient parking never selects another CPU."""
    preferred=declaration(preferred)
    if type(allowed_mask) is not int or allowed_mask<1:raise ValueError("Invalid process affinity mask")
    if any(row["group"]!=0 for row in rows):raise ValueError("Multi-group topology requires a separately declared protocol")
    matching=[row for row in rows if structural(row)==preferred]
    if len(matching)!=1:raise ValueError("Declared CPU is unavailable or its topology differs; no fallback is allowed")
    target=matching[0]
    if not allowed_mask&(1<<target["logical"]):raise ValueError("Declared CPU is outside this process's permitted affinity")
    if target["efficiency_class"]!=max(row["efficiency_class"] for row in rows):raise ValueError("Declared CPU is not in the highest OS-reported performance class")
    if target["allocated"] and not target["allocated_to_process"]:raise ValueError("Declared CPU is reserved for another process")
    return dict(target)

class Processor(C.Structure):
    _fields_=[("group",W.WORD),("number",C.c_ubyte),("reserved",C.c_ubyte)]

class PerformanceCore:
    def __init__(self,preferred):self.preferred=declaration(preferred)

    def __enter__(self):
        if os.name!="nt":raise RuntimeError("This declared CPU protocol requires Windows CPU-set topology")
        self.kernel=C.WinDLL("kernel32",use_last_error=True);k=self.kernel
        k.GetCurrentProcess.restype=W.HANDLE
        k.GetProcessAffinityMask.argtypes=[W.HANDLE,C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)];k.GetProcessAffinityMask.restype=W.BOOL
        k.SetProcessAffinityMask.argtypes=[W.HANDLE,C.c_size_t];k.SetProcessAffinityMask.restype=W.BOOL
        k.GetSystemCpuSetInformation.argtypes=[C.c_void_p,W.ULONG,C.POINTER(W.ULONG),W.HANDLE,W.ULONG];k.GetSystemCpuSetInformation.restype=W.BOOL
        k.GetCurrentProcessorNumberEx.argtypes=[C.POINTER(Processor)];k.GetCurrentProcessorNumberEx.restype=None
        self.handle=k.GetCurrentProcess();self.old_mask=self._mask();self.rows=self._rows()
        selected=select_core(self.rows,self.old_mask,self.preferred);self.mask=1<<selected["logical"]
        self.metadata=dict(protocol=PROTOCOL,selected=structural(selected),mask=self.mask,
                           efficiency_classes=sorted({row["efficiency_class"] for row in self.rows}),logical_processors=len(self.rows),
                           topology=sorted((structural(row) for row in self.rows),key=lambda row:(row["group"],row["logical"],row["id"])),
                           power_settings_changed=False,priority_changed=False)
        self.observations=dict(original_mask=self.old_mask,initial_flags={key:selected[key] for key in ("parked","allocated","allocated_to_process")})
        if not k.SetProcessAffinityMask(self.handle,self.mask):raise C.WinError(C.get_last_error())
        try:
            # Binding the actual process wakes its declared CPU. Verify real
            # execution; do not rewrite metadata or choose an alternate CPU.
            deadline=time.monotonic()+1
            while True:
                processor=self._processor()
                if self._mask()==self.mask and processor==dict(group=0,logical=selected["logical"]):break
                if time.monotonic()>=deadline:raise RuntimeError("Process did not execute on its declared CPU")
                time.sleep(0)
            self.observations["verified_processor"]=self.verify()
            current=select_core(self._rows(),self.mask,self.preferred)
            self.observations["flags_after_pin"]={key:current[key] for key in ("parked","allocated","allocated_to_process")}
        except Exception:
            self._restore();raise
        return self

    def _mask(self):
        actual=C.c_size_t();system=C.c_size_t()
        if not self.kernel.GetProcessAffinityMask(self.handle,C.byref(actual),C.byref(system)):raise C.WinError(C.get_last_error())
        return actual.value

    def _processor(self):
        processor=Processor();self.kernel.GetCurrentProcessorNumberEx(C.byref(processor))
        return dict(group=int(processor.group),logical=int(processor.number))

    def _rows(self):
        length=W.ULONG();k=self.kernel;k.GetSystemCpuSetInformation(None,0,C.byref(length),self.handle,0)
        if not length.value:raise C.WinError(C.get_last_error())
        buffer=C.create_string_buffer(length.value)
        if not k.GetSystemCpuSetInformation(buffer,len(buffer),C.byref(length),self.handle,0):raise C.WinError(C.get_last_error())
        return parse_cpu_sets(buffer.raw[:length.value])

    def verify(self):
        processor=self._processor();actual=self._mask()
        if actual!=self.mask or processor!={"group":0,"logical":self.preferred["logical"]}:raise RuntimeError("Benchmark process/thread left its declared CPU")
        return dict(**processor,mask=actual)

    def _restore(self):
        if not self.kernel.SetProcessAffinityMask(self.handle,self.old_mask):raise C.WinError(C.get_last_error())
        if self._mask()!=self.old_mask:raise RuntimeError("Original process affinity was not restored")

    def __exit__(self,*_):self._restore()
