import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import boto3


def load_application():
    session = boto3.Session(
        aws_access_key_id="offline-test",
        aws_secret_access_key="offline-test",
        region_name="us-east-1",
    )
    with (
        patch("boto3.Session", return_value=session),
        patch("boto3.client", side_effect=session.client),
        patch("botocore.client.BaseClient._make_api_call", side_effect=AssertionError("AWS call in offline test")),
    ):
        spec = importlib.util.spec_from_file_location("support_agent", Path(__file__).parents[1] / "main.py")
        application = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(application)
        return application


class DiscountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = load_application()

    def setUp(self):
        network = patch("botocore.httpsession.URLLib3Session.send", side_effect=AssertionError("Network in offline test"))
        network.start()
        self.addCleanup(network.stop)

    def calculate(self, points=4250, tier="Gold", total=150, category="standard", text_content=False):
        def execute(method, arguments):
            self.assertEqual(method, "executeCode")
            self.assertEqual(arguments["language"], "python")
            self.assertIs(arguments["clearContext"], True)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(compile(arguments["code"], "discount.py", "exec"), {})
            if text_content:
                result = {"content": [{"type": "text", "text": output.getvalue()}]}
            else:
                result = {"structuredContent": {"stdout": output.getvalue(), "exitCode": 0}}
            return {"stream": iter([{"result": result}])}

        with patch.object(self.app, "code_session") as session:
            interpreter = session.return_value.__enter__.return_value
            interpreter.invoke.side_effect = execute
            result = json.loads(self.app.calculate_loyalty_discount(points, tier, total, category))
            session.assert_called_once_with("us-east-1")
            session.return_value.__exit__.assert_called_once()
            self.assertEqual(result["calculation_mode"], "code_interpreter")
            return result

    def test_project_example(self):
        result = self.calculate()
        self.assertEqual(result, {
            "points_redeemed": 4000, "points_discount": 40.0,
            "tier_discount_pct": 10, "tier_discount": 11.0,
            "final_total": 99.0, "total_savings": 51.0,
            "points_earned": 99, "remaining_points": 349,
            "calculation_mode": "code_interpreter",
        })

    def test_redemption_boundaries(self):
        for points, total, redeemed in [(499, 150, 0), (500, 150, 500), (999, 150, 500),
                                        (10000, 9.99, 0), (10000, 10, 500), (10000, 19.99, 500),
                                        (10000, 20, 1000), (10000, 150, 7500), (500, 0, 0)]:
            with self.subTest(points=points, total=total):
                result = self.calculate(points=points, total=total)
                self.assertEqual(result["points_redeemed"], redeemed)
                self.assertLessEqual(result["points_discount"], total / 2)
                self.assertGreaterEqual(result["final_total"], 0)

    def test_tiers_and_earn_rates(self):
        for tier, final in [("Silver", 100.0), ("Gold", 90.0), ("Platinum", 85.0)]:
            for category, rate in [("standard", 1), ("device", 2), ("fresh", 5)]:
                with self.subTest(tier=tier, category=category):
                    result = self.calculate(points=0, tier=tier, total=100, category=category)
                    self.assertEqual(result["final_total"], final)
                    self.assertEqual(result["points_earned"], int(final * rate))
                    self.assertEqual(result["remaining_points"], int(final * rate))

    def test_cent_rounding_and_text_response(self):
        result = self.calculate(points=0, total=10.05, text_content=True)
        self.assertEqual(result["tier_discount"], 1.01)
        self.assertEqual(result["final_total"], 9.04)

    def test_invalid_inputs_never_start_session(self):
        for arguments in [(-1, "Gold", 150), (True, "Gold", 150), (1.5, "Gold", 150),
                          (500, "Unknown", 150), (500, "Gold", -1), (500, "Gold", float("nan")),
                          (500, "Gold", float("inf")), (500, "Gold", True),
                          (500, "Gold", 150, "__import__('os')")]:
            with self.subTest(arguments=arguments), patch.object(self.app, "code_session") as session:
                self.assertIn("error", json.loads(self.app.calculate_loyalty_discount(*arguments)))
                session.assert_not_called()

    def test_session_failure_returns_tier_only_fallback(self):
        with patch.object(self.app, "code_session", side_effect=RuntimeError("private detail")):
            result = json.loads(self.app.calculate_loyalty_discount(4250, "Gold", 150))
        self.assertEqual(result["calculation_mode"], "tier_only_fallback")
        self.assertEqual(result["points_redeemed"], 0)
        self.assertEqual(result["points_earned"], 0)
        self.assertEqual(result["remaining_points"], 4250)
        self.assertEqual(result["final_total"], 135.0)
        self.assertNotIn("private detail", json.dumps(result))

    def test_stream_failures_and_malformed_results_use_fallback(self):
        streams = [[], [{"accessDeniedException": {}}], [{"result": {"isError": True}}],
                   [{"result": {"structuredContent": {"exitCode": 1}}}],
                   [{"result": {"structuredContent": {"stdout": "not json"}}}],
                   [{"result": {"structuredContent": {"stdout": "{}"}}}]]
        for events in streams:
            with self.subTest(events=events), patch.object(self.app, "code_session") as session:
                session.return_value.__enter__.return_value.invoke.return_value = {"stream": iter(events)}
                result = json.loads(self.app.calculate_loyalty_discount(4250, "Gold", 150))
                self.assertEqual(result["calculation_mode"], "tier_only_fallback")
                session.return_value.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
