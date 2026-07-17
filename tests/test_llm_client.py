"""Offline test for the LLM call-rate throttle. No API key/network needed.
Run with: pytest tests/test_llm_client.py
"""
from unittest.mock import patch

from app import llm_client


def test_throttle_sleeps_once_the_per_minute_limit_is_hit():
    llm_client._call_times.clear()
    fake_now = [1000.0]

    def fake_monotonic():
        return fake_now[0]

    sleep_calls = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        fake_now[0] += seconds

    with patch("app.llm_client.time.monotonic", side_effect=fake_monotonic), \
         patch("app.llm_client.time.sleep", side_effect=fake_sleep), \
         patch.object(llm_client.settings, "llm_max_requests_per_minute", 2):
        llm_client._throttle()  # call 1: no sleep
        llm_client._throttle()  # call 2: no sleep, limit now reached
        llm_client._throttle()  # call 3: over the limit, must sleep

    assert sleep_calls == [60.0]
    assert len(llm_client._call_times) == 1  # both prior calls aged out during the wait


if __name__ == "__main__":
    test_throttle_sleeps_once_the_per_minute_limit_is_hit()
    print("OK")
