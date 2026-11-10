"""Partition local wall time without double-counting overlapping own requests."""
import collections
import datetime as dt
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
start=dt.datetime.fromisoformat(json.loads((HERE/'STATE.json').read_text())['created_at']).timestamp()
end=dt.datetime.now(dt.timezone.utc).timestamp()
rows=[json.loads(path.read_text()) for path in (HERE/'requests').glob('*/status.json')]
events=[(start,'boundary',0),(end,'boundary',0)]
for row in rows:
    begin=dt.datetime.strptime(row['started_utc'],'%Y%m%dT%H%M%S.%fZ').replace(tzinfo=dt.timezone.utc).timestamp()
    finish=dt.datetime.fromisoformat(row['completed_at']).timestamp()
    wait=row.get('queue_and_idle_check_seconds')
    acquired=min(begin+wait,finish) if wait is not None else finish
    for kind,left,right in [('waiting',begin,acquired),('running',acquired,finish)]:
        if right>left:events.extend([(left,kind,1),(right,kind,-1)])
counts=collections.Counter();durations=collections.Counter();previous=start
for time,kind,delta in sorted(events):
    time=max(start,min(time,end))
    category='running' if counts['running'] else 'waiting_without_own_running' if counts['waiting'] else 'outside_completed_requests'
    durations[category]+=time-previous
    counts[kind]+=delta;previous=time
assert abs(sum(durations.values())-(end-start))<.01
report={'as_of':dt.datetime.fromtimestamp(end,dt.timezone.utc).isoformat(),
        'elapsed_seconds':end-start,'wall_partition_seconds':dict(durations),
        'request_wait_seconds_sum':sum(row.get('queue_and_idle_check_seconds') or 0 for row in rows),
        'request_running_seconds_sum':sum(row.get('reservation_seconds') or 0 for row in rows),
        'completed_requests':len(rows),'guard_exit_counts':dict(collections.Counter(str(row['guard_exit_code']) for row in rows)),
        'limitations':['Running includes device work, compilation and profiler analysis; it is not hardware utilization.',
                      'Request sums can overlap; wall partition gives own-running precedence and avoids double-counting.',
                      'During an unfinished run, active request time is included in outside_completed_requests until its receipt arrives.',
                      'Different operators and baseline scopes prevent causal comparison with prior three-agent runs.']}
(HERE/'timing-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
