# tests/test_lookup.py
import pandas as pd
import pytest

def test_lookup_happy_path(client, monkeypatch):
    import fastapi_phoenix_agent as mod

    # Small fake dataset
    df = pd.DataFrame(
        {
            "store_id": [1, 2, 1],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "sales": [100, 150, 120],
        }
    )

    # Read parquet -> return our df (no disk access)
    monkeypatch.setattr(mod.pd, "read_parquet", lambda *_a, **_k: df)

    # Force the LLM SQL to a deterministic statement
    monkeypatch.setattr(mod, "generate_sql_query", lambda prompt, cols, table: "SELECT * FROM sales")

    resp = client.post("/lookup", json={"prompt": "show all"})
    assert resp.status_code == 200
    data = resp.json()["result"]
    assert isinstance(data, list)
    assert len(data) == len(df)
    # keys should match
    assert set(data[0].keys()) == {"store_id", "date", "sales"}


def test_lookup_sql_error_bubbles(client, monkeypatch):
    import fastapi_phoenix_agent as mod

    df = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(mod.pd, "read_parquet", lambda *_a, **_k: df)

    # Return a broken SQL
    monkeypatch.setattr(mod, "generate_sql_query", lambda *a, **k: "SELECT * FROM no_such_table")

    resp = client.post("/lookup", json={"prompt": "cause error"})
    # Should become a 500 with detail
    assert resp.status_code == 500
    assert "no_such_table" in resp.json()["detail"]
