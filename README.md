# Customer Support AI Agent with Amazon Bedrock AgentCore

A cloud-deployed customer support agent built with Amazon Bedrock AgentCore. The project combines tool use, retrieval-augmented generation (RAG), cross-session memory, sandboxed computation, and live web browsing in one support workflow.

The agent can track orders, initiate refunds, answer policy and loyalty questions from a knowledge base, remember customer preferences across sessions, calculate loyalty discounts, and retrieve information from live web pages.

## Architecture

```text
Customer Request
      |
      v
Amazon Bedrock AgentCore Runtime
      |
      +--> Strands Agent / Amazon Nova 2 Lite
      |
      +--> MCPClient --> AgentCore Gateway
      |                   |
      |                   +--> API Gateway --> Order Tracking API
      |                   |
      |                   +--> AWS Lambda --> Refund Processor
      |
      +--> Amazon Bedrock Knowledge Base --> RAG
      |
      +--> AgentCore Memory
      |       +--> Semantic Memory
      |       +--> User Preference Memory
      |       +--> Summarization Memory
      |
      +--> AgentCore Code Interpreter --> Loyalty calculations
      |
      +--> AgentCore Browser --> Live web pages
```

The application uses **BedrockAgentCoreApp** at module level, an asynchronous **@app.entrypoint** invocation handler, and **app.run()** as its runtime entry point.

## Implemented Capabilities

### Order tracking

The agent accesses order information through an MCP tool exposed by AgentCore Gateway and backed by API Gateway. Responses are grounded in the tool result rather than generated from model assumptions.

### Refund processing

Refund requests use a separate Gateway target backed by AWS Lambda. The agent can initiate the refund workflow and return the refund identifier, processing status, and expected credit timeline.

### Retrieval-Augmented Generation

The **search_knowledge_base** tool calls the Amazon Bedrock Knowledge Base Retrieve API, joins retrieved chunks into a formatted context string, and provides product, return-policy, and loyalty information to the agent.

### Cross-session memory

AgentCore Memory stores and retrieves customer context across separate conversations. The implementation dynamically discovers strategy IDs and namespace templates, supports both **namespaceTemplates** and the legacy **namespaces** field, and handles actor- and session-scoped namespaces.

### Loyalty discount calculation

The **calculate_loyalty_discount** tool creates a self-contained Python calculation and executes it through AgentCore Code Interpreter. It returns structured values for:

- **points_redeemed**
- **tier_discount_pct**
- **final_total**
- **remaining_points**

A tier-only fallback is available when Code Interpreter is unavailable.

### Live browser access

AgentCore Browser is included in the agent's tools so the deployed agent can retrieve information from live web pages.

## Technology Stack

- Python 3.13
- Amazon Bedrock AgentCore Runtime
- Amazon Bedrock AgentCore Gateway
- Amazon Bedrock AgentCore Memory
- Amazon Bedrock AgentCore Browser
- Amazon Bedrock AgentCore Code Interpreter
- Amazon Bedrock Knowledge Bases
- Model Context Protocol (MCP)
- Strands Agents
- Amazon Nova 2 Lite
- Amazon API Gateway
- AWS Lambda
- Amazon S3
- Docker
- **uv**
- **pytest**

## Project Structure

```text
aws-agentcore-ai-support-agent/
├── README.md
├── screenshots/
│   ├── 01_agentcore_gateway_review.png
│   ├── ...
│   └── 18_test_6_browser_tool.png
└── starter/
    ├── main.py
    ├── Dockerfile
    ├── pyproject.toml
    ├── uv.lock
    ├── .python-version
    └── tests/
        └── test_support_agent.py
```

**starter/main.py** is the primary application implementation. Temporary deployment/migration working directories are intentionally not part of the final project architecture or submission.

## Python Runtime Compatibility

The final AgentCore application targets **Python 3.13**.

During deployment testing, the application was initially packaged with Python 3.14. A fresh runtime invocation could succeed, while a subsequent request in the same runtime session could fail inside the Starlette/AnyIO thread-pool path used by AgentCore's synchronous **/ping** endpoint. The installed **bedrock-agentcore** package explicitly classified Python versions through 3.13.

The project was therefore moved to Python 3.13 while keeping the relevant AgentCore, Starlette, AnyIO, and Uvicorn versions unchanged. The complete offline suite passed under Python 3.13, and repeated deployed-runtime invocations subsequently completed successfully.

## Setup

### Prerequisites

You need:

- An AWS account with access to the required Bedrock and AgentCore services
- AWS credentials configured outside the repository
- **uv**
- AgentCore CLI
- Docker/container build capability as required by your deployment workflow

Never commit AWS access keys, secret keys, session tokens, or other credentials.

### Install dependencies

From **starter/**:

```bash
uv sync --python 3.13
```

The repository includes **.python-version** and declares Python **>=3.13**.

### Configuration

The application requires identifiers/endpoints for its AWS resources, including:

- AWS region
- AgentCore Gateway URL
- Knowledge Base ID
- AgentCore Memory ID

For production systems, environment variables or a managed configuration/secrets service should be preferred over environment-specific values embedded in source.

## Run Tests

From **starter/**:

```bash
uv run --python 3.13 --with pytest pytest -q
```

Validated result:

```text
27 passed
68 subtests passed
```

A Pydantic deprecation warning originating from the installed AgentCore SDK was present but did not cause test failures.

## Deploy and Invoke

Deploy using the AgentCore deployment workflow configured for the project. Once the Runtime reports **READY**, invoke the deployed agent with the AgentCore CLI.

Example:

```bash
agentcore invoke '{"prompt": "Can you track order ORD-001?", "customer_id": "CUST-123", "session_id": "t1"}'
```

The final deployed acceptance run completed all six required scenarios successfully.

## Acceptance Tests

### Test 1 — Order Tracking

```bash
agentcore invoke '{"prompt": "Can you track order ORD-001?", "customer_id": "CUST-123", "session_id": "t1"}'
```

Observed result: the deployed agent returned **SHIPPED**, UPS, tracking number **TRK987654321**, and the estimated delivery date.

### Test 2 — Refund Processing

```bash
agentcore invoke '{"prompt": "I want to return my Kindle Paperwhite (ORD-002). Please initiate a refund.", "customer_id": "CUST-123", "session_id": "t2"}'
```

Observed result: a refund ID was returned with **APPROVED** status and a **3–5 business days** credit timeline.

### Test 3 — Knowledge Base (RAG)

```bash
agentcore invoke '{"prompt": "What are the benefits of the Platinum loyalty tier?", "customer_id": "CUST-123", "session_id": "t3"}'
```

Observed result: the agent retrieved the expected Knowledge Base information, including free same-day shipping, a 15% discount, and priority customer support.

### Test 4 — Long-Term Memory

Session A:

```bash
agentcore invoke '{"prompt": "Hi, I am Jane. I prefer concise responses.", "customer_id": "CUST-123", "session_id": "s-A"}'
```

After allowing time for memory extraction, Session B used a different session ID:

```bash
agentcore invoke '{"prompt": "Do you remember my name and communication preference?", "customer_id": "CUST-123", "session_id": "s-B"}'
```

Observed result: the agent recalled **Jane** and her preference for **concise responses**, demonstrating cross-session recall for the same customer.

### Test 5 — Loyalty Discount Calculation

```bash
agentcore invoke '{"prompt": "I am a Gold member with 4250 points. Calculate my discount on a $150 standard order.", "customer_id": "CUST-123", "session_id": "t5"}'
```

Observed result:

- Points redeemed: 4,000
- Tier discount: 10%
- Final total: $99.00
- Remaining points: 350

### Test 6 — Browser Tool

```bash
agentcore invoke '{"prompt": "Go to https://www.udacity.com and tell me the page title.", "customer_id": "CUST-123", "session_id": "t6"}'
```

Observed result: the Browser tool accessed the live Udacity site and returned the page title **“Learn the Latest Tech Skills; Advance Your Career | Udacity”**.

> The supplied project instructions contained an inconsistency: the executable Test 6 prompt specifies Udacity, while its expected-result sentence mentions Amazon.com. The implementation was validated using the executable Udacity test command.

## Screenshots and Evidence

The **screenshots/** directory documents the project from infrastructure configuration through final deployed acceptance testing.

| # | Screenshot | What it demonstrates |
| ---: | --- | --- |
| 01 | **01_agentcore_gateway_review.png** | AgentCore Gateway configuration review. |
| 02 | **02_api_gateway_curl_validation.png** | Successful external API Gateway validation with curl. |
| 03 | **03_agentcore_gateway_target_ready.png** | AgentCore Gateway target in ready state. |
| 04 | **04_harness_order_tracker_success.png** | Successful order-tracker tool execution through the AgentCore Harness. |
| 05 | **05_harness_order_tracker_agent_trace_success.png** | Agent trace showing successful order-tracker tool use. |
| 06 | **06_harness_get_return_label_success.png** | Successful return-label workflow validation in the Harness. |
| 07 | **07_harness_initiate_refund_agent_trace_success.png** | Agent trace for the Lambda-backed refund workflow. |
| 08 | **08_harness_return_label_agent_trace_success.png** | Agent trace for return-label processing. |
| 09 | **09_harness_initiate_refund_success.png** | Successful refund-processing tool result in the Harness. |
| 10 | **10_knowledge_base_retrieval_success.png** | Successful Amazon Bedrock Knowledge Base retrieval. |
| 11 | **11_memory_created_success.png** | AgentCore Memory resource created and active. |
| 12 | **12_agentcore_runtime_ready.png** | Deployed AgentCore Runtime in **READY** state. |
| 13 | **13_test_1_order_tracking.png** | Final deployed Test 1: order tracking. |
| 14 | **14_test_2_refund_processing.png** | Final deployed Test 2: refund processing. |
| 15 | **15_test_3_knowledge_base_rag.png** | Final deployed Test 3: RAG / Knowledge Base. |
| 16 | **16_test_4_long_term_memory.png** | Final deployed Test 4: cross-session memory recall. |
| 17 | **17_test_5_loyalty_discount_calculation.png** | Final deployed Test 5: Code Interpreter loyalty calculation. |
| 18 | **18_test_6_browser_tool.png** | Final deployed Test 6: live Browser tool. |

### Final Acceptance Evidence

#### Test 1 — Order Tracking

![Test 1 — Order Tracking](screenshots/13_test_1_order_tracking.png)

#### Test 2 — Refund Processing

![Test 2 — Refund Processing](screenshots/14_test_2_refund_processing.png)

#### Test 3 — Knowledge Base (RAG)

![Test 3 — Knowledge Base](screenshots/15_test_3_knowledge_base_rag.png)

#### Test 4 — Long-Term Memory

![Test 4 — Long-Term Memory](screenshots/16_test_4_long_term_memory.png)

#### Test 5 — Loyalty Discount Calculation

![Test 5 — Loyalty Discount](screenshots/17_test_5_loyalty_discount_calculation.png)

#### Test 6 — Browser Tool

![Test 6 — Browser Tool](screenshots/18_test_6_browser_tool.png)

## Engineering and Responsible-AI Considerations

The agent is designed to use authoritative tools for operational facts rather than allowing the language model to invent order, refund, or policy information. Customer ownership should be validated before exposing order-specific information, and tool responses should be treated as external data rather than trusted instructions.

For a production deployment, additional controls should include:

- least-privilege IAM policies for Runtime, Memory, Knowledge Base, Browser, Gateway, and Lambda access;
- authentication and authorization before customer-specific operations;
- idempotency protection for side-effectful actions such as refunds;
- structured validation of tool inputs and outputs;
- encryption and data-retention policies for customer memory;
- PII minimization and explicit retention/consent policies;
- CloudWatch metrics, logs, tracing, alarms, and tool-failure monitoring;
- prompt-injection defenses for retrieved and browsed content;
- latency and token-cost monitoring;
- rate limits and concurrency controls;
- evaluation for hallucinations, unsafe tool use, retrieval quality, and memory leakage across customers.

## Project Reflection

A detailed implementation reflection covering the AgentCore integration,
the runtime compatibility issue encountered during deployment, and production
considerations is available in [REFLECTION.md](REFLECTION.md).

## Cleanup

After assessment and evidence collection are complete, remove cloud resources that are no longer needed to avoid unnecessary cost.

Typical cleanup includes:

1. Destroy the AgentCore deployment using the project's supported teardown workflow.
2. Delete the AgentCore Gateway and Memory resources.
3. Delete the Bedrock Knowledge Base.
4. Delete the underlying OpenSearch Serverless collection if it was created for the Knowledge Base.
5. Empty and delete the Knowledge Base S3 bucket.
6. Delete the API Gateway API.
7. Delete the Lambda functions.
8. Remove project-specific IAM roles only after confirming they are no longer used.

Do **not** destroy resources until final validation, screenshots, and submission requirements are complete.

## Status

**Completed:** all six required deployed acceptance scenarios passed, including Gateway-backed order/refund tools, RAG, cross-session memory, Code Interpreter calculation, and live Browser access.
