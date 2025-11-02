# tests/test_root.py
def test_root_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # keys exist (values may be None or strings depending on your env)
    assert "phoenix_ui" in body
    assert "phoenix_traces_endpoint" in body
