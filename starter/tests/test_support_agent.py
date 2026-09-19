import contextlib
import importlib.util
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
import uuid

import boto3
from strands import Agent, tool
from strands.types import PaginatedList


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


class OrchestrationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = load_application()
        cls.memory_hook = cls.app.MemoryHook

    def setUp(self):
        self.mocks = {}
        for name in ("MemoryHook", "memory_client", "AgentCoreBrowser", "MCPClient", "Agent", "streamable_http_client"):
            replacement = patch.object(self.app, name)
            self.mocks[name] = replacement.start()
            self.addCleanup(replacement.stop)
        network = patch("botocore.httpsession.URLLib3Session.send", side_effect=AssertionError("Network in offline test"))
        network.start()
        self.addCleanup(network.stop)
        self.memory = self.mocks["memory_client"]
        self.memory.list_events.return_value = []
        self.browser = self.mocks["AgentCoreBrowser"].return_value
        self.browser.close.return_value = {"status": "success"}
        self.gateway = self.mocks["MCPClient"].return_value.__enter__.return_value
        self.gateway.list_tools_sync.return_value = PaginatedList([Mock(name="order_tool")])
        self.agent = self.mocks["Agent"].return_value
        self.agent.invoke_async = AsyncMock(return_value=SimpleNamespace(message={"content": [{"text": "Tool-grounded answer"}]}))
        self.payload = {"prompt": "Track ORD-001", "customer_id": "CUST-123", "session_id": "s1"}

    async def test_wires_tools_hooks_and_current_gateway(self):
        result = await self.app.invoke(self.payload)
        self.assertEqual(result, "Tool-grounded answer")
        self.mocks["MemoryHook"].assert_called_once_with("CUST-123", "s1", self.memory, self.app.MEMORY_ID)
        self.memory.list_events.assert_called_once_with(
            memory_id=self.app.MEMORY_ID, actor_id="CUST-123", session_id="s1", max_results=20, include_payload=True,
        )
        factory = self.mocks["MCPClient"].call_args.args[0]
        factory()
        self.mocks["streamable_http_client"].assert_called_once_with(self.app.GATEWAY_URL)
        self.mocks["AgentCoreBrowser"].assert_called_once_with(region="us-east-1")
        options = self.mocks["Agent"].call_args.kwargs
        self.assertEqual(options["tools"][:3], [self.app.search_knowledge_base, self.app.calculate_loyalty_discount, self.browser.browser])
        self.assertEqual(len(options["tools"]), 4)
        self.assertEqual(options["hooks"], [self.mocks["MemoryHook"].return_value])
        self.assertIn("CUST-123", options["system_prompt"])
        self.assertIsNone(options["callback_handler"])
        self.agent.invoke_async.assert_awaited_once_with(self.payload["prompt"])
        self.mocks["MCPClient"].return_value.__exit__.assert_called_once()
        self.browser.close.assert_called_once()

    async def test_mcp_context_stays_open_during_agent_invocation(self):
        async def answer(_):
            self.mocks["MCPClient"].return_value.__exit__.assert_not_called()
            return SimpleNamespace(message={"content": [{"text": "Ready"}]})
        self.agent.invoke_async.side_effect = answer
        self.assertEqual(await self.app.invoke(self.payload), "Ready")

    async def test_gateway_pagination(self):
        first, second = Mock(), Mock()
        self.gateway.list_tools_sync.side_effect = [PaginatedList([first], token="next"), PaginatedList([second])]
        await self.app.invoke(self.payload)
        self.assertEqual(self.mocks["Agent"].call_args.kwargs["tools"][-2:], [first, second])
        self.assertEqual(self.gateway.list_tools_sync.call_args_list[1].kwargs, {"pagination_token": "next"})

    async def test_restores_session_history_in_timestamp_order(self):
        def event(day, query, answer):
            return {"eventTimestamp": datetime(2026, 9, day, tzinfo=timezone.utc), "payload": [
                {"conversational": {"role": "USER", "content": {"text": query}}},
                {"conversational": {"role": "ASSISTANT", "content": {"text": answer}}},
                {"blob": "not a conversation"},
            ]}
        self.memory.list_events.return_value = [event(19, "Second", "Second reply"), event(18, "First", "First reply")]
        await self.app.invoke(self.payload)
        history = self.mocks["Agent"].call_args.kwargs["messages"]
        self.assertEqual([message["content"][0]["text"] for message in history], ["First", "First reply", "Second", "Second reply"])

    async def test_anonymous_request_does_not_access_customer_memory(self):
        await self.app.invoke({"prompt": "What can you help with?"})
        self.mocks["MemoryHook"].assert_not_called()
        self.memory.list_events.assert_not_called()
        options = self.mocks["Agent"].call_args.kwargs
        self.assertEqual(options["hooks"], [])
        self.assertEqual(options["messages"], [])
        self.assertIn("No customer ID", options["system_prompt"])

    async def test_missing_session_uses_fresh_uuid(self):
        await self.app.invoke({"prompt": "Hello", "customer_id": "CUST-123"})
        first = self.mocks["MemoryHook"].call_args.args[1]
        uuid.UUID(first)
        await self.app.invoke({"prompt": "Hello", "customer_id": "CUST-123"})
        self.assertNotEqual(first, self.mocks["MemoryHook"].call_args.args[1])

    async def test_invalid_requests_do_not_start_services(self):
        for payload in [None, [], {}, {"prompt": " "}, {"prompt": 1},
                        {"prompt": "Hi", "customer_id": "../other"},
                        {"prompt": "Hi", "customer_id": ""}, {"prompt": "Hi", "session_id": 42}]:
            with self.subTest(payload=payload):
                result = await self.app.invoke(payload)
                self.assertIsInstance(result, str)
                self.mocks["MCPClient"].assert_not_called()
                self.mocks["AgentCoreBrowser"].assert_not_called()
                self.memory.list_events.assert_not_called()

    async def test_empty_tools_and_repeated_tokens_fail_safely(self):
        for pages in [[PaginatedList([])], [PaginatedList([Mock()], token="same"), PaginatedList([Mock()], token="same")]]:
            with self.subTest(pages=pages):
                self.gateway.list_tools_sync.side_effect = pages
                result = await self.app.invoke(self.payload)
                self.assertIn("service is unavailable", result)
                self.mocks["Agent"].assert_not_called()

    async def test_gateway_and_model_failures_cleanup_without_exposing_details(self):
        for component in (self.gateway.list_tools_sync, self.agent.invoke_async):
            with self.subTest(component=component):
                component.side_effect = RuntimeError("private service detail")
                result = await self.app.invoke(self.payload)
                self.assertNotIn("private service detail", result)
                self.assertIn("refund", result)
                self.browser.close.assert_called()
                self.mocks["MCPClient"].return_value.__exit__.assert_called()
                component.side_effect = None

    async def test_memory_failure_does_not_start_agent(self):
        self.memory.list_events.side_effect = RuntimeError("memory unavailable")
        self.assertIn("service is unavailable", await self.app.invoke(self.payload))
        self.mocks["Agent"].assert_not_called()

    async def test_empty_response_and_cleanup_failure(self):
        self.agent.invoke_async.return_value = SimpleNamespace(message={"content": []})
        self.assertIn("could not produce a response", await self.app.invoke(self.payload))
        self.agent.invoke_async.return_value = SimpleNamespace(message={"content": [{"text": "Answer"}]})
        self.browser.close.side_effect = RuntimeError("cleanup failure")
        self.assertEqual(await self.app.invoke(self.payload), "Answer")

    async def test_real_agent_tool_cycle_and_memory_hooks_with_offline_model(self):
        @tool
        def browser_stub() -> str:
            """Stand in for the browser without opening a browser session."""
            return "unused"

        @tool
        def gateway_stub() -> str:
            """Stand in for a Gateway tool without connecting to a server."""
            return "unused"

        self.browser.browser = browser_stub
        self.gateway.list_tools_sync.return_value = PaginatedList([gateway_stub])
        self.mocks["MemoryHook"].side_effect = self.memory_hook
        self.memory.get_memory_strategies.return_value = [
            {"type": "SEMANTIC", "namespaceTemplates": ["cs_agent/{actorId}/facts"]},
        ]
        self.memory.retrieve_memories.return_value = [{"content": {"text": "Prefers concise replies"}}]
        self.mocks["Agent"].side_effect = Agent
        calls = []

        async def stream(messages, *args, **kwargs):
            calls.append(messages)
            yield {"messageStart": {"role": "assistant"}}
            if len(calls) == 1:
                self.assertIn("Customer Context:", messages[-1]["content"][0]["text"])
                yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "lookup-1", "name": "search_knowledge_base"}}}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"query":"electronics return policy"}'}}}}
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "tool_use"}}
            else:
                self.assertIn("15 days", json.dumps(messages[-1]))
                yield {"contentBlockDelta": {"delta": {"text": "Electronics may be returned within 15 days."}}}
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "end_turn"}}
            yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}}

        with (
            patch.object(self.app.model, "stream", side_effect=stream),
            patch.object(self.app._bedrock_runtime, "retrieve", return_value={"retrievalResults": [{"content": {"text": "Electronics: 15 days"}}]}) as retrieve,
        ):
            payload = {**self.payload, "prompt": "What is the electronics return policy?"}
            result = await self.app.invoke(payload)
        self.assertEqual(result, "Electronics may be returned within 15 days.")
        self.assertEqual(len(calls), 2)
        retrieve.assert_called_once_with(knowledgeBaseId=self.app.KB_ID, retrievalQuery={"text": "electronics return policy"})
        self.memory.create_event.assert_called_once_with(
            memory_id=self.app.MEMORY_ID, actor_id="CUST-123", session_id="s1",
            messages=[(payload["prompt"], "USER"), (result, "ASSISTANT")],
        )


if __name__ == "__main__":
    unittest.main()
