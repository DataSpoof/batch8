# tests/test_agent_run.py
import types
import json

class _Msg:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

class _Choice:
    def __init__(self, message):
        self.message = message

class _Resp:
    def __init__(self, message):
        self.choices = [_Choice(message)]

def test_agent_run_no_tool_calls(client, monkeypatch):
    import fastapi_phoenix_agent as mod

    # Fake OpenAI client that always returns a final message with no tool calls
    class _FakeChatCompletions:
        def create(self, model, messages, tools=None):
            return _Resp(_Msg("final answer", tool_calls=[]))

    class _FakeChat:
        completions = _FakeChatCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(mod, "client", _FakeClient())

    resp = client.post("/agent/run", json={"messages": [{"role": "user", "content": "Hi"}]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["result"] == "final answer"


def test_agent_run_with_tool_call(client, monkeypatch):
    import fastapi_phoenix_agent as mod

    # 1st model call -> tool call
    tool_call_id = "call_1"
    tool_name = "lookup_sales_data"
    tool_args = {"prompt": "show all"}

    class FnObj:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = json.dumps(arguments)

    class ToolCall:
        def __init__(self, id, fn):
            self.id = id
            self.function = fn

    first_message = _Msg(
        content=None,
        tool_calls=[ToolCall(tool_call_id, FnObj(tool_name, tool_args))]
    )

    # 2nd model call -> final answer
    second_message = _Msg("tool result consumed, final")

    calls = {"count": 0}

    class _FakeChatCompletions:
        def create(self, model, messages, tools=None):
            # first call returns a tool call, second call returns final
            if calls["count"] == 0:
                calls["count"] += 1
                return _Resp(first_message)
            else:
                return _Resp(second_message)

    class _FakeChat:
        completions = _FakeChatCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(mod, "client", _FakeClient())

    # Stub lookup_sales_data tool to avoid file I/O
    monkeypatch.setattr(mod, "lookup_sales_data", lambda prompt: '[{"ok":1}]')
    # IMPORTANT: refresh cached mapping so the agent uses the stubbed function
    mod.tool_implementations["lookup_sales_data"] = mod.lookup_sales_data

    resp = client.post("/agent/run", json={"messages": [{"role": "user", "content": "Run an analysis"}]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["result"] == "tool result consumed, final"
