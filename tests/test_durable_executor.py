from business.durable_executor import DurableWriteExecutor, idempotency_key_for
from business.schemas import ToolExecutionResult, ToolExecutionStatus
from jobs import JobStatus, JobStore


class FakeInner:
    def __init__(self):
        self.calls = 0

    def execute(self, tool, arguments, user_text, customer_id):
        self.calls += 1
        if tool == "get_outstanding_balance":
            return ToolExecutionResult(tool=tool, status=ToolExecutionStatus.SUCCESS, success=True,
                                       user_message="balance", data={"balance": 1200})
        if tool == "blocked_write":
            return ToolExecutionResult(tool="record_promise_to_pay", status=ToolExecutionStatus.BLOCKED,
                                       success=False, user_message="blocked")
        return ToolExecutionResult(tool=tool, status=ToolExecutionStatus.SUCCESS, success=True,
                                   user_message="recorded", data={"record_id": 7})


def test_write_tool_enqueues_job_and_reads_do_not(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    ex = DurableWriteExecutor(FakeInner(), store, scope="app")

    r = ex.execute("get_outstanding_balance", {}, "balance?", "C1")
    assert "delivery" not in r.data and store.stats()["QUEUED"] == 0

    r = ex.execute("record_promise_to_pay", {"amount": 500, "date": "2026-09-05"}, "I'll pay", "C1")
    assert r.success and r.data["delivery"]["status"] == "QUEUED"
    assert r.data["delivery"]["deduplicated"] is False
    job = store.get(r.data["delivery"]["job_id"])
    assert job.type == "business_write" and job.payload["tool"] == "record_promise_to_pay"
    assert job.status is JobStatus.QUEUED


def test_same_turn_repeated_is_one_job(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    ex = DurableWriteExecutor(FakeInner(), store, scope="app")
    a = ex.execute("schedule_callback", {"time": "5pm"}, "call me", "C1")
    b = ex.execute("schedule_callback", {"time": "5pm"}, "call me", "C1")
    assert a.data["delivery"]["job_id"] == b.data["delivery"]["job_id"]
    assert b.data["delivery"]["deduplicated"] is True
    assert store.stats()["QUEUED"] == 1


def test_key_is_customer_scoped_and_argument_order_independent():
    k1 = idempotency_key_for("app", "C1", "open_dispute", {"a": 1, "b": 2})
    k2 = idempotency_key_for("app", "C1", "open_dispute", {"b": 2, "a": 1})
    k3 = idempotency_key_for("app", "C2", "open_dispute", {"a": 1, "b": 2})
    assert k1 == k2 and k1 != k3


def test_blocked_write_never_enqueues(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    ex = DurableWriteExecutor(FakeInner(), store, scope="app")
    r = ex.execute("blocked_write", {}, "x", "C1")
    assert not r.success and store.stats()["QUEUED"] == 0
