"""IBM hardware path, exercised offline with a fake backend and a stand-in service."""

import numpy as np
import pytest

pytest.importorskip("qiskit_ibm_runtime")

from qstock import ibm  # noqa: E402


class FakeService:
    """Mimics the parts of QiskitRuntimeService that qstock.ibm uses."""

    def __init__(self):
        from qiskit_ibm_runtime.fake_provider import FakeFez

        self.backend_obj = FakeFez()
        self.jobs = {}

    def least_busy(self, **_):
        return self.backend_obj

    def backend(self, _name):
        return self.backend_obj

    def job(self, job_id):
        return self.jobs[job_id]

    def usage(self):
        return {"usage_remaining_ru": 9.5}


def _toy():
    pmf = np.array([0.1, 0.2, 0.3, 0.4])
    cond = np.array([0.0, 0.0, 1.0, 1.0])
    return pmf, cond  # exact a = 0.7


def test_preview_tracks_exact_value():
    pmf, cond = _toy()
    plan, res = ibm.preview(pmf, cond, shots=4000, include_grover=True)
    assert plan.exact["k=0"] == pytest.approx(0.7)
    k0 = next(r for r in res["rows"] if r["circuit"] == "k=0")
    assert abs(k0["corrected"] - 0.7) < 0.05
    assert set(plan.labels) == {"k=0", "k=1", "cal0", "cal1"}


def test_submit_and_fetch_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(ibm, "JOBS_FILE", tmp_path / "jobs.json")
    svc = FakeService()
    from qiskit_ibm_runtime import SamplerV2

    real_run = SamplerV2.run

    def run_and_remember(self, pubs, shots=None):
        job = real_run(self, pubs, shots=shots)
        svc.jobs[job.job_id()] = job
        return job

    monkeypatch.setattr(SamplerV2, "run", run_and_remember)
    pmf, cond = _toy()
    rec = ibm.submit(svc, pmf, cond, shots=2000, include_grover=False, meta={"ticker": "TEST"})
    assert ibm.load_jobs()[0]["job_id"] == rec["job_id"]
    svc.jobs[rec["job_id"]].result()  # wait, as a real queue would make us
    status, res, _ = ibm.fetch(svc, rec)
    assert status == "DONE"
    assert abs(res["estimate_corrected"] - 0.7) < 0.06
    assert ibm.remaining_minutes(svc) == pytest.approx(9.5)


def test_readout_correction_formula():
    class R:
        def __init__(self, ones, shots):
            from types import SimpleNamespace

            counts = {"1": ones, "0": shots - ones}
            self.data = SimpleNamespace(c=SimpleNamespace(get_counts=lambda: counts))

    # true a = 0.5, e0 = 0.02, e1 = 0.06  ->  measured = 0.5*0.94 + 0.5*0.02 = 0.48
    res = ibm.analyse(["k=0", "cal0", "cal1"], [R(480, 1000), R(20, 1000), R(940, 1000)], {"k=0": 0.5}, 1.0)
    assert res["rows"][0]["corrected"] == pytest.approx(0.5)
