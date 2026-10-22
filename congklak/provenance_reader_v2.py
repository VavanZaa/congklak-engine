"""Strict completed provenance scope; no writer, engines or replay searches."""
from . import provenance_job as job, ownership_implementation_common as stage

def completed(path=job.DEFAULT):
    result=job.completed(path)
    expected=dict(stage.baseline()['inputs'])
    expected.update(job.module().plan()['inputs'])
    expected.update(stage.common.pins([stage.FOLDER/'baseline.json',*job.SOURCES]))
    stage.need(result['identity']['inputs']==expected,'Provenance complete input coverage differs')
    return result
