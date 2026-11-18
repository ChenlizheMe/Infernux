"""Conversation delivery counts actual IDs and clears them with their owner."""
import unittest

from infernux_automation.delivery import ReportDelivery


class Queries:
    def __init__(self):
        self.revision = 1
        self.callbacks = []
        self.stopped = []

    def add_stop_listener(self, callback):
        self.callbacks.append(callback)

    def snapshot(self, identity):
        return {"id": identity, "revision": self.revision, "status": "ready", "value": self.revision}

    def stop(self, identity):
        self.stopped.append(identity)
        for callback in self.callbacks:
            callback(identity)
        return True


class ReportDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.queries = Queries()
        self.delivery = ReportDelivery(self.queries)

    def test_new_report_does_not_count_an_old_non_adjacent_user_turn(self):
        self.delivery.add("old", "chat", reports=2, mode="turns", owner="session")
        self.assertEqual(self.delivery.consume("chat", "A")[0]["remaining"], 1)
        self.assertEqual(self.delivery.consume("chat", "B")[0]["remaining"], 0)
        self.delivery.add("new", "chat", reports=3, mode="turns", owner="session")
        self.assertEqual(self.delivery.consume("chat", "A"), [])
        self.assertEqual(self.delivery.consume("chat", "B"), [])
        self.assertEqual(self.delivery.consume("chat", "C")[0]["remaining"], 2)

    def test_prompt_ids_before_subscription_are_not_future_user_turns(self):
        self.delivery.consume("chat", "A")
        self.delivery.consume("chat", "B")
        self.delivery.add("new", "chat", reports=2, mode="turns", owner="session")
        self.assertEqual(self.delivery.consume("chat", "A"), [])
        self.assertEqual(self.delivery.consume("chat", "B"), [])
        self.assertEqual(self.delivery.consume("chat", "C")[0]["remaining"], 1)

    def test_an_unidentified_prompt_hook_can_read_changes_after_a_known_turn(self):
        self.delivery.add("query", "chat", reports=3, mode="changes", owner="session")
        self.delivery.add("turn-query", "chat", reports=3, mode="turns", owner="session")
        self.delivery.consume("chat", "A")
        self.queries.revision = 2
        changed = self.delivery.consume("chat", changes_only=True)
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["remaining"], 1)
        self.assertEqual(changed[0]["observation"]["revision"], 2)
        self.assertEqual(self.delivery.consume("chat", changes_only=True), [])

    def test_live_turn_ids_are_not_evicted_and_session_close_releases_them(self):
        self.delivery.add("query", "chat", reports=1100, mode="turns", owner="session")
        for index in range(1050):
            self.delivery.consume("chat", f"turn-{index}")
        # More unrelated consumers than the old LRU capacity must not evict the
        # active conversation's current cycle or its non-adjacent seen IDs.
        for index in range(1050):
            self.delivery.consume(f"other-{index}", "other-turn")
        self.assertEqual(self.delivery.consume("chat", "turn-0"), [])
        self.assertEqual(self.delivery.consume("chat"), [])
        self.delivery.close_owner("session")
        self.assertNotIn("chat", self.delivery._cycles)
        self.assertNotIn("chat", self.delivery._seen_cycles)
        self.assertNotIn("chat", self.delivery._consumer_owners)
        self.delivery.add("new-query", "chat", reports=2, mode="turns", owner="new-session")
        self.assertEqual(self.delivery.consume("chat", "turn-0")[0]["remaining"], 1)

    def test_closing_one_owner_preserves_the_other_conversation_and_shared_consumer(self):
        self.delivery.add("one", "same-chat", reports=3, mode="turns", owner="one")
        self.delivery.add("two", "same-chat", reports=3, mode="turns", owner="two")
        self.delivery.add("other", "other-chat", reports=3, mode="turns", owner="two")
        self.delivery.consume("same-chat", "A")
        self.delivery.consume("other-chat", "A")
        self.delivery.close_owner("one")
        self.assertEqual(self.delivery.consume("same-chat", "A"), [])
        self.assertEqual(self.delivery.consume("other-chat", "A"), [])
        self.assertEqual(self.delivery.consume("same-chat", "B")[0]["remaining"], 1)
        self.assertEqual(self.delivery.consume("other-chat", "B")[0]["remaining"], 1)
        self.delivery.close_owner("two")
        self.assertEqual(self.delivery._consumer_owners, {})
        self.assertEqual(self.delivery._seen_cycles, {})

    def test_pending_change_can_be_delivered_once_when_ready_in_the_same_turn(self):
        self.queries.revision = 0
        self.delivery.add("query", "chat", reports=2, mode="changes", owner="session")
        self.assertEqual(self.delivery.consume("chat", "A"), [])
        self.queries.revision = 1
        self.assertEqual(self.delivery.consume("chat", "A")[0]["remaining"], 1)
        self.assertEqual(self.delivery.consume("chat", "A"), [])


if __name__ == "__main__":
    unittest.main()
