# tests/test_analyze.py
def test_analyze_endpoint(client, monkeypatch):
    import fastapi_phoenix_agent as mod

    monkeypatch.setattr(mod, "analyze_sales_data", lambda prompt, data: f"Analysis OK for: {prompt}")

    payload = {"prompt": "Top stores?", "data": '[{"store_id":1,"sales":100}]'}
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200
    assert resp.json()["analysis"] == "Analysis OK for: Top stores?"
