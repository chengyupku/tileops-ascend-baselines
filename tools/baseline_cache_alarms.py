#!/usr/bin/env python3
"""Seven absolute campaign milestones, delivered only while Codex is idle.

Reuses session_alarm's kernel-lock owner discovery and idempotent queue/start.
No turn/start, turn/steer, stale-home addressing, or active-turn injection.
"""
import argparse
import contextlib
import datetime
import fcntl
import json
from pathlib import Path
import time
import uuid

import session_alarm as alarm

MILESTONES = [300, 600, 1800, 3600, 7200, 14400, 28800]


def extend_recurring_schedule(state):
    """After one delivery, schedule the next original clock boundary.

    A long active turn may delay delivery past several boundaries. Those are
    coalesced, never replayed as back-to-back wakeups when the user returns.
    Delivery remains separate from the inspection's checked_at acknowledgement.
    """
    interval = state.get('interval_seconds')
    if not interval or not state.get('milestones') or state.get('pending'):
        return
    last = state['milestones'][-1]
    if last.get('delivered_at') is None:
        return
    due = last['due_at'] + interval
    missed = max(0, int((last['delivered_at'] - due) // interval) + 1)
    due += missed * interval
    state['milestones'].append({'after_seconds': due-state['start_at'], 'due_at': due,
                                'coalesced_boundaries': missed})


def inspect_schedule(state, api, now, save):
    obs = api.observe()
    state['last_observed_at'] = now
    state['last_session_status'] = obs['status']
    extend_recurring_schedule(state)
    pending = state.get('pending')
    if pending:
        prior_phase = pending['phase']
        consumed = (prior_phase in ('starting', 'queued') and obs['turn'] != pending['anchor_turn']
                    and not any(q['clientUserMessageId'] == pending['id'] for q in api.queued()))
        # If a real user started a turn, cancel the queued alarm and wait for idle.
        if pending['phase'] != 'off':
            alarm.step(pending, obs, api, now, save)
        if pending['phase'] == 'off':
            if pending.get('wake_turn') or 'consumed' in pending.get('reason', '') or consumed:
                if consumed and not pending.get('wake_turn'):
                    pending['delivery_evidence'] = 'queue submission consumed and session advanced; RPC response unavailable, inspection acknowledgement still required'
                for i in pending['milestones']:
                    state['milestones'][i]['delivered_at'] = now
                    state['milestones'][i]['alarm_id'] = pending['id']
                state.setdefault('deliveries', []).append(dict(pending))
            state.pop('pending', None)
        save()
        return
    # Do not even enqueue a reminder during an active repair.
    if obs['status'] != 'idle' or obs.get('direct') is not True:
        save()
        return
    due = [i for i, m in enumerate(state['milestones']) if m['due_at'] <= now and 'delivered_at' not in m]
    if not due:
        save()
        return
    if state.get('interval_seconds'):
        labels = ', '.join(datetime.datetime.fromtimestamp(state['milestones'][i]['due_at'], datetime.timezone.utc).isoformat() for i in due)
    else:
        labels = ', '.join(str(state['milestones'][i]['after_seconds']//60)+'min' for i in due)
    message = state['message'] + '\n本次到期节点：' + labels + '。闹钟投递不等于巡检完成；请记录真实检查时间和行动。'
    state['pending'] = {'id': str(uuid.uuid4()), 'thread': state['thread'], 'phase': 'armed',
        'due_at': now, 'anchor_turn': obs['turn'], 'after_seconds': 0, 'message': message,
        'milestones': due}
    save()
    alarm.step(state['pending'], obs, api, now, save)
    # Persist delivery now; next tick only acknowledges the completed one-shot.


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--init-thread')
    p.add_argument('--start', type=float)
    p.add_argument('--message-file', type=Path)
    p.add_argument('--interval-seconds', type=int, help='Optional recurring idle-only inspection interval')
    a = p.parse_args()
    with a.schedule.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if a.init_thread:
            if a.schedule.exists():
                raise FileExistsError('Do not reset a started campaign clock')
            start = a.start if a.start is not None else time.time()
            if a.interval_seconds is not None and a.interval_seconds <= 0:
                raise ValueError('interval must be positive')
            intervals = [a.interval_seconds] if a.interval_seconds else MILESTONES
            state = {'thread': alarm.thread_id(a.init_thread), 'start_at': start,
                'milestones': [{'after_seconds': s, 'due_at': start+s} for s in intervals],
                'message': a.message_file.read_text()}
            if a.interval_seconds:
                state['interval_seconds'] = a.interval_seconds
            alarm.put(a.schedule, state)
            return
        state = alarm.read(a.schedule)
        if state.get('disabled'):
            return
        save = lambda: alarm.put(a.schedule, state)
        try:
            with contextlib.closing(alarm.RPC(alarm.resolve_owner(state['thread']), state['thread'])) as api:
                inspect_schedule(state, api, time.time(), save)
            state.pop('last_error', None)
        except Exception as exc:
            state.update(last_error=type(exc).__name__+': '+str(exc), error_at=time.time())
        save()


if __name__ == '__main__':
    main()
