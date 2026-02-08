"""
Unit tests for NL-to-SQL query engine

Tests SQL validation, schema context building, result formatting,
and query execution logic. Ollama API calls are mocked to avoid
requiring a running Ollama server for testing.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai.nl_to_sql import (
    NLToSQLEngine,
    NLToSQLError,
    SQLValidationError,
    QueryExecutionError,
)


# =============================================================================
# Helper: Create engine with mocked dependencies
# =============================================================================


@pytest.fixture
def engine():
    """Create an NLToSQLEngine with mocked dependencies (no Ollama or DB needed)."""
    with patch.object(NLToSQLEngine, "__init__", lambda self, *args, **kwargs: None):
        eng = NLToSQLEngine.__new__(NLToSQLEngine)
        eng.config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "clickstream_warehouse",
                "username": "dataeng",
                "password": "dataeng123",
            },
            "ai": {
                "model": "llama3",
                "ollama_url": "http://localhost:11434",
                "temperature": 0.0,
                "max_result_rows": 100,
                "query_timeout_seconds": 30,
            },
        }
        eng.ollama_url = "http://localhost:11434"
        eng.model = "llama3"
        eng.schema_context = "TABLE: dim_users\n  - user_id: varchar(50)"
        eng.system_prompt = "You are a SQL expert."
        return eng


# =============================================================================
# SQL Validation Tests
# =============================================================================


class TestSQLValidation:
    """Tests for SQL validation safety logic."""

    def test_valid_select_query(self, engine):
        is_valid, error = engine.validate_sql("SELECT * FROM dim_users")
        assert is_valid is True
        assert error == ""

    def test_valid_select_with_join(self, engine):
        sql = (
            "SELECT u.user_id, s.session_id "
            "FROM dim_users u JOIN fact_sessions s ON u.user_id = s.user_id"
        )
        is_valid, error = engine.validate_sql(sql)
        assert is_valid is True

    def test_valid_cte_query(self, engine):
        sql = (
            "WITH top_users AS (SELECT user_id FROM dim_users LIMIT 10) "
            "SELECT * FROM top_users"
        )
        is_valid, error = engine.validate_sql(sql)
        assert is_valid is True

    def test_valid_select_with_subquery(self, engine):
        sql = "SELECT * FROM (SELECT user_id, total_revenue FROM dim_users) AS sub"
        is_valid, error = engine.validate_sql(sql)
        assert is_valid is True

    def test_valid_select_with_aggregation(self, engine):
        sql = (
            "SELECT user_segment, COUNT(*) AS cnt "
            "FROM dim_users GROUP BY user_segment ORDER BY cnt DESC"
        )
        is_valid, error = engine.validate_sql(sql)
        assert is_valid is True

    def test_rejects_insert(self, engine):
        sql = "INSERT INTO dim_users (user_id) VALUES ('hacker')"
        is_valid, error = engine.validate_sql(sql)
        assert is_valid is False
        assert "INSERT" in error.upper() or "must start with SELECT" in error.lower()

    def test_rejects_update(self, engine):
        sql = "UPDATE dim_users SET user_segment = 'Hacked'"
        is_valid, error = engine.validate_sql(sql)
        assert is_valid is False

    def test_rejects_delete(self, engine):
        sql = "DELETE FROM dim_users"
        is_valid, error = engine.validate_sql(sql)
        assert is_valid is False

    def test_rejects_drop(self, engine):
        sql = "DROP TABLE dim_users"
        is_valid, error = engine.validate_sql(sql)
        assert is_valid is False

    def test_rejects_alter(self, engine):
        sql = "ALTER TABLE dim_users ADD COLUMN hacked BOOLEAN"
        is_valid, error = engine.validate_sql(sql)
        assert is_valid is False

    def test_rejects_truncate(self, engine):
        sql = "TRUNCATE TABLE dim_users"
        is_valid, error = engine.validate_sql(sql)
        assert is_valid is False

    def test_rejects_multiple_statements(self, engine):
        sql = "SELECT 1; DROP TABLE dim_users"
        is_valid, error = engine.validate_sql(sql)
        assert is_valid is False
        assert "Multiple" in error or "Forbidden" in error

    def test_rejects_grant(self, engine):
        sql = "GRANT ALL ON dim_users TO public"
        is_valid, error = engine.validate_sql(sql)
        assert is_valid is False

    def test_keyword_in_string_literal_is_ok(self, engine):
        sql = "SELECT * FROM dim_users WHERE user_segment = 'delete me'"
        is_valid, error = engine.validate_sql(sql)
        assert is_valid is True

    def test_empty_sql_rejected(self, engine):
        is_valid, error = engine.validate_sql("")
        assert is_valid is False

    def test_whitespace_only_rejected(self, engine):
        is_valid, error = engine.validate_sql("   ")
        assert is_valid is False


# =============================================================================
# SQL Generation Tests (mocked Ollama API)
# =============================================================================


class TestSQLGeneration:
    """Tests for Ollama API SQL generation (mocked)."""

    @patch("ai.nl_to_sql.requests.post")
    def test_generate_sql_returns_string(self, mock_post, engine):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "SELECT * FROM dim_users LIMIT 10"}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        sql = engine.generate_sql("Show me all users")
        assert isinstance(sql, str)
        assert "SELECT" in sql

    @patch("ai.nl_to_sql.requests.post")
    def test_strips_markdown_fences(self, mock_post, engine):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "```sql\nSELECT * FROM dim_users\n```"}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        sql = engine.generate_sql("Show me all users")
        assert "```" not in sql
        assert sql == "SELECT * FROM dim_users"

    @patch("ai.nl_to_sql.requests.post")
    def test_strips_trailing_semicolon(self, mock_post, engine):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "SELECT * FROM dim_users;"}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        sql = engine.generate_sql("Show me all users")
        assert not sql.endswith(";")

    @patch("ai.nl_to_sql.requests.post")
    def test_extracts_sql_from_explanation(self, mock_post, engine):
        """Test that SQL is extracted when model includes explanation text."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "content": "Here is the query you need:\nSELECT * FROM dim_users LIMIT 10"
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        sql = engine.generate_sql("Show me all users")
        assert sql.startswith("SELECT")

    @patch("ai.nl_to_sql.requests.post")
    def test_handles_connection_error(self, mock_post, engine):
        import requests as req
        mock_post.side_effect = req.ConnectionError("Connection refused")

        with pytest.raises(NLToSQLError, match="Cannot connect to Ollama"):
            engine.generate_sql("Show me all users")

    @patch("ai.nl_to_sql.requests.post")
    def test_handles_timeout_error(self, mock_post, engine):
        import requests as req
        mock_post.side_effect = req.Timeout("Request timed out")

        with pytest.raises(NLToSQLError, match="timed out"):
            engine.generate_sql("Show me all users")

    @patch("ai.nl_to_sql.requests.post")
    def test_handles_generic_api_error(self, mock_post, engine):
        mock_post.side_effect = Exception("Something broke")

        with pytest.raises(NLToSQLError, match="Ollama API error"):
            engine.generate_sql("Show me all users")


# =============================================================================
# Result Formatting Tests
# =============================================================================


class TestResultFormatting:
    """Tests for query result formatting."""

    def test_empty_dataframe(self, engine):
        df = pd.DataFrame()
        result = engine.format_results(df)
        assert "No results" in result

    def test_none_dataframe(self, engine):
        result = engine.format_results(None)
        assert "No results" in result

    def test_small_results_full_display(self, engine):
        df = pd.DataFrame({
            "user_id": ["user_001", "user_002", "user_003"],
            "segment": ["High Value", "Converted", "New/Inactive"],
        })
        result = engine.format_results(df)
        assert "3 row(s)" in result
        assert "user_001" in result
        assert "user_003" in result

    def test_large_results_truncated(self, engine):
        df = pd.DataFrame({
            "user_id": [f"user_{i:03d}" for i in range(100)],
            "value": range(100),
        })
        result = engine.format_results(df)
        assert "100 row(s)" in result
        assert "more rows" in result

    def test_includes_column_count(self, engine):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = engine.format_results(df)
        assert "3 column(s)" in result


# =============================================================================
# End-to-End Ask Pipeline Tests (mocked)
# =============================================================================


class TestAskPipeline:
    """Tests for the full ask() pipeline."""

    @patch("ai.nl_to_sql.requests.post")
    def test_successful_ask(self, mock_post, engine):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "content": "SELECT user_segment, COUNT(*) AS cnt FROM dim_users GROUP BY user_segment"
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        mock_df = pd.DataFrame({
            "user_segment": ["High Value", "Converted"],
            "cnt": [50, 200],
        })
        with patch.object(engine, "execute_query", return_value=mock_df):
            result = engine.ask("How many users per segment?")

        assert result["error"] is None
        assert result["sql"] is not None
        assert result["results_df"] is not None
        assert result["formatted_output"] is not None

    @patch("ai.nl_to_sql.requests.post")
    def test_ask_with_api_error(self, mock_post, engine):
        mock_post.side_effect = Exception("API down")

        result = engine.ask("Show me users")
        assert result["error"] is not None
        assert "generation failed" in result["error"].lower()

    @patch("ai.nl_to_sql.requests.post")
    def test_ask_with_invalid_sql(self, mock_post, engine):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "DROP TABLE dim_users"}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = engine.ask("Delete all users")
        assert result["error"] is not None
        assert "validation failed" in result["error"].lower()

    @patch("ai.nl_to_sql.requests.post")
    def test_ask_with_execution_error(self, mock_post, engine):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "SELECT * FROM nonexistent_table"}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        with patch.object(
            engine,
            "execute_query",
            side_effect=QueryExecutionError("Table not found"),
        ):
            result = engine.ask("Show me bad table")

        assert result["error"] is not None
        assert "execution failed" in result["error"].lower()


# =============================================================================
# Schema Context Tests
# =============================================================================


class TestSchemaContext:
    """Tests for schema context building."""

    def test_schema_from_file_fallback(self):
        """Test that schema can be built from sql/schema.sql file."""
        with patch.object(
            NLToSQLEngine, "__init__", lambda self, *args, **kwargs: None
        ):
            eng = NLToSQLEngine.__new__(NLToSQLEngine)
            schema = eng._build_schema_from_file()

            assert len(schema) > 0
            assert "dim_users" in schema
            assert "fact_events" in schema
            assert "fact_sessions" in schema

    def test_schema_context_is_not_empty(self, engine):
        """Verify the engine has non-empty schema context."""
        assert engine.schema_context is not None
        assert len(engine.schema_context) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
