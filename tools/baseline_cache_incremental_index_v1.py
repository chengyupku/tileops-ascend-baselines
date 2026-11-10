"""CPU-only projections of the append-only cache journal; never changes receipts."""
import json
import os
from pathlib import Path
class CacheIndex:
    def __init__(self,path):
        self.path=Path(path);self.file_id=None;self.offset=0;self.stamp=None;self.ids=set();self.best={};self.records=0
    def refresh(self):
        if not self.path.exists():return self
        with self.path.open('rb') as f:
            st=os.fstat(f.fileno());ident=(st.st_dev,st.st_ino);stamp=(st.st_size,st.st_mtime_ns)
            if ident!=self.file_id or st.st_size<self.offset or (self.stamp and st.st_size==self.offset and stamp!=self.stamp):
                self.offset=0;self.ids.clear();self.best.clear();self.records=0
            self.file_id=ident;f.seek(self.offset)
            while True:
                start=f.tell();line=f.readline()
                if not line:break
                if not line.endswith(b'\n'):f.seek(start);break
                row=json.loads(line);self.ids.add(row['task_id']);self.records+=1
                key=(row['key'][1],row['key'][2],row['cohort_key'],row['comparison_metric'])
                latency=row['timing']['latency_ms'] if key[3]=='device_latency_ms' else row['end_to_end_mean_ms']
                if key not in self.best or latency<self.best[key]['latency_ms']:
                    self.best[key]={'op':key[0],'case_id':key[1],'cohort':key[2],'metric':key[3],'library':row['key'][0],'latency_ms':latency,'cache_key':row['cache_key'],'coverage':'best_among_successfully_measured_candidates_only'}
                self.offset=f.tell()
            self.stamp=stamp
        return self
    def task_rows(self):
        self.refresh();return ({'task_id':i} for i in self.ids)
    def best_rows(self):
        self.refresh();return list(self.best.values())
_indices={}
def cache_index(path):
    key=str(Path(path).resolve())
    if key not in _indices:_indices[key]=CacheIndex(key)
    return _indices[key]
