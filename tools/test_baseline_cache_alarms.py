import copy
import unittest
from baseline_cache_alarms import inspect_schedule


class Fake:
    def __init__(self, status='idle', turn='one'):
        self.status, self.turn = status, turn
        self.queue, self.added, self.started = [], 0, 0
    def observe(self):
        return {'status': self.status, 'turn': self.turn, 'direct': True}
    def queued(self):
        return self.queue[:]
    def add(self, a):
        self.added += 1
        self.queue.append({'id': 'q', 'clientUserMessageId': a['id']})
        return 'q'
    def delete(self, qid):
        self.queue = [q for q in self.queue if q['id'] != qid]
    def start(self, qid):
        self.started += 1
        self.queue = []
        self.status, self.turn = 'active', 'alarmturn'
        return {'turn': {'id': self.turn}}


class TestAlarm(unittest.TestCase):
    def state(self):
        return {'thread': 't', 'message': 'inspect', 'milestones': [
            {'after_seconds': 300, 'due_at': 300}, {'after_seconds': 600, 'due_at': 600}]}
    def test_active_is_never_queued(self):
        s, api = self.state(), Fake('active')
        inspect_schedule(s, api, 1000, lambda: None)
        self.assertEqual((api.added, api.started), (0, 0))
    def test_due_absolute_times_and_single_delivery(self):
        s, api = self.state(), Fake()
        inspect_schedule(s, api, 299, lambda: None)
        self.assertEqual(api.added, 0)
        inspect_schedule(s, api, 301, lambda: None)
        inspect_schedule(s, api, 302, lambda: None)
        self.assertEqual(api.started, 1)
        self.assertIn('delivered_at', s['milestones'][0])
        self.assertNotIn('delivered_at', s['milestones'][1])
        api.status, api.turn = 'idle', 'a-different-home-resumed'
        inspect_schedule(s, api, 601, lambda: None)
        inspect_schedule(s, api, 602, lambda: None)
        self.assertEqual(api.started, 2)
        self.assertIn('delivered_at', s['milestones'][1])
    def test_coalesce_overdue_during_long_repair(self):
        s, api = self.state(), Fake()
        inspect_schedule(s, api, 900, lambda: None)
        inspect_schedule(s, api, 901, lambda: None)
        self.assertEqual(api.started, 1)
        self.assertTrue(all('delivered_at' in m for m in s['milestones']))
    def test_existing_user_queue_wins(self):
        s, api = self.state(), Fake()
        api.queue = [{'id': 'real-user', 'clientUserMessageId': 'user'}]
        inspect_schedule(s, api, 900, lambda: None)
        self.assertEqual(api.added, 0)
        self.assertEqual(api.started, 0)
    def test_consumed_wake_with_lost_rpc_response_is_not_reissued(self):
        s, api = self.state(), Fake()
        start = api.start
        def missing_turn(qid):
            start(qid)
            return {}
        api.start = missing_turn
        with self.assertRaises(KeyError):
            inspect_schedule(s, api, 301, lambda: None)
        inspect_schedule(s, api, 302, lambda: None)
        self.assertEqual(api.started, 1)
        self.assertIn('delivered_at', s['milestones'][0])

    def test_recurring_wakes_at_each_four_hour_boundary_only_when_idle(self):
        s={'thread':'t','message':'inspect','start_at':0,'interval_seconds':14400,
           'milestones':[{'after_seconds':14400,'due_at':14400}]}
        api=Fake()
        inspect_schedule(s,api,14400,lambda:None)
        inspect_schedule(s,api,14401,lambda:None)
        inspect_schedule(s,api,28801,lambda:None)
        self.assertEqual(api.started,1)  # Still active: do not steer or enqueue.
        self.assertEqual(s['milestones'][-1]['due_at'],28800)
        self.assertNotIn('checked_at',s['milestones'][0])  # Delivery is not inspection.
        api.status,api.turn='idle','resumed-in-another-home'
        inspect_schedule(s,api,28802,lambda:None)
        inspect_schedule(s,api,28803,lambda:None)
        self.assertEqual(api.started,2)

    def test_recurring_long_busy_interval_does_not_replay_missed_wakes(self):
        s={'thread':'t','message':'inspect','start_at':0,'interval_seconds':14400,
           'milestones':[{'after_seconds':14400,'due_at':14400}]}
        api=Fake('active')
        inspect_schedule(s,api,60000,lambda:None)
        self.assertEqual(api.added,0)
        api.status='idle'
        inspect_schedule(s,api,60001,lambda:None)
        inspect_schedule(s,api,60002,lambda:None)
        api.status='idle'
        inspect_schedule(s,api,60003,lambda:None)
        self.assertEqual(api.started,1)
        self.assertEqual(s['milestones'][-1]['due_at'],72000)
        self.assertEqual(s['milestones'][-1]['coalesced_boundaries'],3)


if __name__ == '__main__':
    unittest.main()
