# tests/test_visualize.py
def test_visualize_endpoint(client, monkeypatch):
    import fastapi_phoenix_agent as mod

    monkeypatch.setattr(mod, "generate_visualization", lambda data, goal: "print('chart ok')")

    payload = {"data": '[{"x": 1, "y": 2}]', "visualization_goal": "Line chart of y over x"}
    resp = client.post("/visualize", json=payload)
    assert resp.status_code == 200
    assert resp.json()["code"] == "print('chart ok')"
