"""Stable declared CPU selection and actual own-process affinity restoration."""
import ctypes as C
from ctypes import wintypes as W
import struct
import unittest
from congklak.timing_v2 import PerformanceCore,select_core,parse_cpu_sets,structural

PREFERRED=dict(id=258,group=0,logical=2,core=2,cache=0,numa=0,efficiency_class=1)

def synthetic():
    return [dict(id=256+i,group=0,logical=i,core=i//2*2,cache=0,numa=0,efficiency_class=1 if i<4 else 0,
                 parked=False,allocated=False,allocated_to_process=False) for i in range(8)]

def affinity():
    kernel=C.WinDLL("kernel32",use_last_error=True);kernel.GetCurrentProcess.restype=W.HANDLE
    kernel.GetProcessAffinityMask.argtypes=[W.HANDLE,C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)];kernel.GetProcessAffinityMask.restype=W.BOOL
    kernel.SetProcessAffinityMask.argtypes=[W.HANDLE,C.c_size_t];kernel.SetProcessAffinityMask.restype=W.BOOL
    handle=kernel.GetCurrentProcess()
    def get():
        process=C.c_size_t();system=C.c_size_t()
        if not kernel.GetProcessAffinityMask(handle,C.byref(process),C.byref(system)):raise C.WinError(C.get_last_error())
        return process.value
    def put(mask):
        if not kernel.SetProcessAffinityMask(handle,mask):raise C.WinError(C.get_last_error())
    return get,put

class TimingV2Acceptance(unittest.TestCase):
    def test_01_synthetic_parking_never_changes_declared_cpu(self):
        rows=synthetic();selected=select_core(rows,255,PREFERRED)
        rows[2]["parked"]=True;rows[0]["parked"]=True;rows[1]["parked"]=True
        parked=select_core(rows,255,PREFERRED)
        self.assertTrue(parked["parked"]);self.assertEqual(structural(selected),structural(parked));self.assertEqual(parked["logical"],2)
        rows[2]["allocated"]=True;rows[2]["allocated_to_process"]=True
        self.assertEqual(structural(select_core(rows,4,PREFERRED)),PREFERRED)

    def test_02_unavailable_disallowed_reserved_or_changed_cpu_refused(self):
        rows=synthetic()
        with self.assertRaises(ValueError):select_core([r for r in rows if r["id"]!=258],255,PREFERRED)
        with self.assertRaises(ValueError):select_core(rows,1,PREFERRED)
        rows[2]["allocated"]=True
        with self.assertRaises(ValueError):select_core(rows,255,PREFERRED)
        rows[2]["allocated"]=False;rows[2]["core"]=99
        with self.assertRaises(ValueError):select_core(rows,255,PREFERRED)
        rows=synthetic();rows[4]["efficiency_class"]=2
        with self.assertRaises(ValueError):select_core(rows,255,PREFERRED)
        with self.assertRaises(ValueError):PerformanceCore({**PREFERRED,"parked":False})
        with self.assertRaises(ValueError):PerformanceCore({**PREFERRED,"logical":True})
        with self.assertRaises(ValueError):PerformanceCore({**PREFERRED,"group":1})

    def test_03_binary_parser_variable_records_and_flags(self):
        record=bytearray(40);struct.pack_into("<II",record,0,40,0)
        struct.pack_into("<IHBBBBBB",record,8,258,0,2,2,0,0,1,7)
        unknown=struct.pack("<II8x",16,99)
        row=parse_cpu_sets(unknown+record)[0]
        self.assertEqual(structural(row),PREFERRED);self.assertTrue(all(row[k] for k in ("parked","allocated","allocated_to_process")))
        for bad in (b"",b"123",struct.pack("<II",7,0),struct.pack("<II",50,0),struct.pack("<II8x",16,0)):
            with self.assertRaises(ValueError):parse_cpu_sets(bad)

    def test_04_actual_pin_restore_exception_and_startup_mask_independence(self):
        get,put=affinity();original=get()
        with PerformanceCore(preferred=PREFERRED) as cpu:
            first=cpu.metadata;self.assertEqual(cpu.verify(),dict(group=0,logical=2,mask=4));self.assertEqual(cpu.observations["original_mask"],original)
            self.assertNotIn("original_mask",cpu.metadata);self.assertFalse(set(cpu.metadata["selected"])&{"parked","allocated","allocated_to_process"})
        self.assertEqual(get(),original)
        # A genuine different inherited process mask is observational only;
        # both starts really execute on the exact same declared processor.
        try:
            put(4)
            with PerformanceCore(preferred=PREFERRED) as cpu:
                self.assertEqual(cpu.metadata,first);self.assertEqual(cpu.observations["original_mask"],4);cpu.verify()
            self.assertEqual(get(),4)
            with self.assertRaisesRegex(RuntimeError,"fixture exception"):
                with PerformanceCore(preferred=PREFERRED) as cpu:
                    cpu.verify();raise RuntimeError("fixture exception")
            self.assertEqual(get(),4)
        finally:put(original)
        self.assertEqual(get(),original)

if __name__=="__main__":unittest.main()
