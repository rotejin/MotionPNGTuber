"""tools/audit.py fails on every advisory whose impact has not been checked.

Before, every torch advisory and every pillow / setuptools advisory without a fixed
version (``fix_versions == []``) was left out without anybody reading it.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "tools", "audit.py")
_spec = importlib.util.spec_from_file_location("motionpngtuber_tools_audit", _PATH)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

REVIEWED_PILLOW = "GHSA-5xm2-6676-qcp9"  # in REVIEWED["pillow"]
REVIEWED_TORCH = "GHSA-53q9-r3pm-6pq6"  # in REVIEWED["torch"]


def vuln(package, vid, *, aliases=(), fixes=(), version="1.0"):
    return {
        "dependency": {"name": package, "version": version},
        "id": vid,
        "aliases": list(aliases),
        "fix_versions": list(fixes),
        "summary": "test advisory",
        "link": "https://example.invalid/" + vid,
    }


class AuditTests(unittest.TestCase):
    def run_audit(self, vulnerabilities, *, adverse=(), stdout=None, which="uv"):
        report = {
            "summary": {"audited_packages": 66},
            "vulnerabilities": vulnerabilities,
            "adverse_statuses": list(adverse),
        }
        result = SimpleNamespace(stdout=json.dumps(report) if stdout is None else stdout, stderr="")
        out = io.StringIO()
        with (
            mock.patch.object(audit.shutil, "which", return_value=which),
            mock.patch.object(audit.subprocess, "run", return_value=result),
            contextlib.redirect_stdout(out),
        ):
            rc = audit.main()
        return rc, out.getvalue()

    def test_reviewed_advisories_that_cannot_be_fixed_pass(self):
        rc, out = self.run_audit([
            vuln("pillow", REVIEWED_PILLOW, fixes=["13.0.0"], version="12.3.0"),
            vuln("torch", REVIEWED_TORCH, fixes=["2.6.0"], version="1.13.1+cu117"),
            vuln("setuptools", "GHSA-xmj3-9gf7-h52w", fixes=["84.1.0"], version="69.5.1"),
        ])
        self.assertEqual(rc, 0, out)
        self.assertIn("OK", out)

    def test_new_advisory_in_a_held_back_package_fails(self):
        cases = {
            "torch, any fix": vuln("torch", "GHSA-new0-torc-h000", fixes=["2.12.0"]),
            "torch, no fix yet": vuln("torch", "PYSEC-2099-1"),
            "pillow, fixed only in 13": vuln("pillow", "GHSA-new0-pill-ow00", fixes=["13.1.0"]),
            "pillow, no fix yet": vuln("pillow", "GHSA-new1-pill-ow00"),
            "setuptools, no fix yet": vuln("setuptools", "GHSA-new0-setu-p000"),
        }
        for name, v in cases.items():
            with self.subTest(name):
                rc, out = self.run_audit([v])
                self.assertEqual(rc, 1, out)
                self.assertIn(v["id"], out)
                self.assertIn("REVIEWED", out)

    def test_a_reviewed_advisory_is_found_through_its_aliases(self):
        # uv may report the same advisory under another id (GHSA / PYSEC / CVE).
        rc, out = self.run_audit([
            vuln("pillow", "PYSEC-2099-77", aliases=["CVE-2099-1", REVIEWED_PILLOW], fixes=["13.0.0"]),
        ])
        self.assertEqual(rc, 0, out)

    def test_reviewed_id_of_another_package_does_not_count(self):
        # torch is held back, but this id was checked for Pillow, not for torch.
        rc, out = self.run_audit([vuln("torch", REVIEWED_PILLOW)])
        self.assertEqual(rc, 1, out)

    def test_reviewed_advisory_with_an_installable_fix_fails(self):
        # For example a fix back-ported to Pillow 12.
        rc, out = self.run_audit([vuln("pillow", REVIEWED_PILLOW, fixes=["12.4.0", "13.0.0"])])
        self.assertEqual(rc, 1, out)
        self.assertIn("12.4.0", out)

    def test_other_packages_fail(self):
        rc, out = self.run_audit([vuln("requests", "GHSA-req0-uest-s000", fixes=["2.40.0"])])
        self.assertEqual(rc, 1, out)

    def test_adverse_status_fails(self):
        rc, out = self.run_audit([], adverse=[{"package": "foo", "status": "yanked"}])
        self.assertEqual(rc, 1, out)

    def test_unreadable_report_and_missing_uv_fail(self):
        rc, out = self.run_audit([], stdout="not json")
        self.assertEqual(rc, 1, out)
        rc, out = self.run_audit([], which=None)
        self.assertEqual(rc, 1, out)
        self.assertIn("uv", out)

    def test_reviewed_ids_look_like_advisory_ids(self):
        for package, ids in audit.REVIEWED.items():
            self.assertIn(package, audit.HELD_BACK)
            for vid in ids:
                self.assertRegex(vid, r"^(GHSA(-[23456789cfghjmpqrvwx]{4}){3}|PYSEC-\d{4}-\d+)$", vid)


if __name__ == "__main__":
    unittest.main()
