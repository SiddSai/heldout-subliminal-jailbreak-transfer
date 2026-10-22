from pathlib import Path

from sdvt.io import append_csv, stable_hash


def test_hash_is_stable() -> None:
    assert stable_hash("same") == stable_hash("same")
    assert stable_hash("same") != stable_hash("different")


def test_csv_schema_cannot_drift(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    append_csv(ledger, [{"id": "a", "decision": "keep"}], ["id", "decision"])
    try:
        append_csv(ledger, [{"id": "b", "other": "x"}], ["id", "other"])
    except ValueError as error:
        assert "schema drift" in str(error)
    else:
        raise AssertionError("Expected schema drift to fail")
