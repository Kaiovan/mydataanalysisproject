"""
Natural Language to SQL Query Engine using Ollama (Local LLM)

Converts plain English questions about e-commerce clickstream data
into valid PostgreSQL queries, executes them read-only, and returns
formatted results.

Uses Ollama to run a local LLM (e.g., llama3, mistral, codellama)
for free with no API key required.
"""

import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import psycopg2
import requests
import yaml
from dotenv import load_dotenv

warnings.filterwarnings("ignore")


# =============================================================================
# EXCEPTIONS
# =============================================================================


class NLToSQLError(Exception):
    """Base exception for NL-to-SQL engine."""
    pass


class SQLValidationError(NLToSQLError):
    """Raised when generated SQL fails safety validation."""
    pass


class QueryExecutionError(NLToSQLError):
    """Raised when SQL execution fails."""
    pass


class SchemaContextError(NLToSQLError):
    """Raised when schema context cannot be built."""
    pass


# =============================================================================
# CORE ENGINE
# =============================================================================


class NLToSQLEngine:
    """
    Natural Language to SQL query engine using Ollama (local LLM).

    Converts plain English questions about e-commerce clickstream data
    into valid PostgreSQL queries, executes them read-only, and returns
    formatted results.

    Uses Ollama running locally — completely free, no API key needed.

    Safety features:
    - SQL validation rejects dangerous statements (INSERT, DROP, etc.)
    - Database connection is set to read-only mode
    - Query timeout prevents runaway queries
    - Auto-LIMIT prevents huge result sets
    """

    # SQL keywords that are NEVER allowed in generated queries
    FORBIDDEN_KEYWORDS = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
        "CREATE", "GRANT", "REVOKE", "EXEC", "EXECUTE", "CALL",
        "COPY", "VACUUM", "REINDEX", "CLUSTER",
    ]

    # Only these keywords can start a query
    ALLOWED_START_KEYWORDS = ["SELECT", "WITH"]

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the NL-to-SQL engine.

        Args:
            config_path: Path to config.yaml. Defaults to project root config.
        """
        if config_path is None:
            config_path = str(
                Path(__file__).parent.parent.parent / "config" / "config.yaml"
            )

        self.config = self._load_config(config_path)
        self.ollama_url = self.config["ai"]["ollama_url"]
        self.model = self.config["ai"]["model"]
        self._check_ollama_connection()
        self.schema_context = self._build_schema_context()
        self.system_prompt = self._build_system_prompt()

    def _load_config(self, config_path: str) -> Dict:
        """
        Load database and AI config from config.yaml.

        Args:
            config_path: Path to config file

        Returns:
            Dictionary with configuration settings
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Set defaults for AI config if not present
        if "ai" not in config:
            config["ai"] = {}

        ai_defaults = {
            "model": "llama3",
            "ollama_url": "http://localhost:11434",
            "temperature": 0.0,
            "max_result_rows": 100,
            "query_timeout_seconds": 30,
        }

        for key, value in ai_defaults.items():
            if key not in config["ai"]:
                config["ai"][key] = value

        return config

    def _check_ollama_connection(self):
        """
        Check if Ollama is running and the model is available.
        """
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            resp.raise_for_status()
            available_models = [
                m["name"].split(":")[0]
                for m in resp.json().get("models", [])
            ]
            if available_models:
                print(f"  Ollama connected. Available models: {', '.join(available_models)}")
            else:
                print(f"  Ollama connected but no models found.")
                print(f"  Pull a model with: ollama pull {self.model}")
        except requests.ConnectionError:
            print(f"  WARNING: Cannot connect to Ollama at {self.ollama_url}")
            print(f"  Make sure Ollama is running: ollama serve")
            print(f"  Then pull a model: ollama pull {self.model}")
        except Exception as e:
            print(f"  WARNING: Ollama check failed: {e}")

    def _get_db_connection(self) -> psycopg2.extensions.connection:
        """
        Create a read-only PostgreSQL connection using config settings.

        Returns:
            psycopg2 connection object configured for read-only access
        """
        db_config = self.config["database"]

        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            database=db_config["database"],
            user=db_config["username"],
            password=db_config["password"],
        )

        # Set read-only mode and query timeout
        conn.set_session(readonly=True, autocommit=True)

        timeout_ms = self.config["ai"]["query_timeout_seconds"] * 1000
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = '{timeout_ms}'")

        return conn

    def _build_schema_context(self) -> str:
        """
        Build schema context by introspecting the live database.

        Queries information_schema to get all tables, views, and their columns.
        Falls back to parsing sql/schema.sql if the database is unavailable.

        Returns:
            Formatted string describing the database schema
        """
        try:
            return self._build_schema_from_db()
        except Exception:
            print("  Database not available, falling back to schema.sql file...")
            return self._build_schema_from_file()

    def _build_schema_from_db(self) -> str:
        """Introspect live database for schema information."""
        db_config = self.config["database"]

        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            database=db_config["database"],
            user=db_config["username"],
            password=db_config["password"],
        )

        try:
            with conn.cursor() as cur:
                # Get all tables and views
                cur.execute("""
                    SELECT table_name, table_type
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_type, table_name
                """)
                tables = cur.fetchall()

                # Get columns for each table
                cur.execute("""
                    SELECT table_name, column_name, data_type,
                           is_nullable, character_maximum_length
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    ORDER BY table_name, ordinal_position
                """)
                columns = cur.fetchall()
        finally:
            conn.close()

        # Build schema text
        schema_lines = []
        columns_by_table = {}
        for table_name, col_name, data_type, nullable, max_len in columns:
            if table_name not in columns_by_table:
                columns_by_table[table_name] = []
            type_str = data_type
            if max_len:
                type_str = f"{data_type}({max_len})"
            null_str = "" if nullable == "YES" else " NOT NULL"
            columns_by_table[table_name].append(f"  - {col_name}: {type_str}{null_str}")

        for table_name, table_type in tables:
            kind = "TABLE" if "TABLE" in table_type else "VIEW"
            schema_lines.append(f"\n{kind}: {table_name}")
            if table_name in columns_by_table:
                schema_lines.extend(columns_by_table[table_name])

        return "\n".join(schema_lines)

    def _build_schema_from_file(self) -> str:
        """Parse sql/schema.sql as fallback when database is unavailable."""
        schema_path = Path(__file__).parent.parent.parent / "sql" / "schema.sql"

        if not schema_path.exists():
            raise SchemaContextError(
                f"Cannot build schema context: database unavailable and "
                f"{schema_path} not found"
            )

        with open(schema_path, "r") as f:
            schema_sql = f.read()

        # Extract CREATE TABLE and CREATE VIEW statements as context
        lines = []
        lines.append("-- Schema extracted from sql/schema.sql --")

        # Find table/view definitions
        for match in re.finditer(
            r"CREATE\s+(?:TABLE|OR\s+REPLACE\s+VIEW|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
            schema_sql,
            re.IGNORECASE,
        ):
            obj_name = match.group(1)
            lines.append(f"\nOBJECT: {obj_name}")

        # Extract column definitions from CREATE TABLE blocks
        for match in re.finditer(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\((.*?)\);",
            schema_sql,
            re.IGNORECASE | re.DOTALL,
        ):
            table_name = match.group(1)
            body = match.group(2)
            lines.append(f"\nTABLE: {table_name}")
            for col_match in re.finditer(
                r"^\s+(\w+)\s+([\w()]+(?:\s*\(\s*\d+(?:,\s*\d+)?\s*\))?)",
                body,
                re.MULTILINE,
            ):
                col_name = col_match.group(1)
                col_type = col_match.group(2)
                if col_name.upper() not in (
                    "PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT",
                    "CREATE", "INDEX",
                ):
                    lines.append(f"  - {col_name}: {col_type}")

        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for the LLM with schema context and instructions.

        Returns:
            Complete system prompt string
        """
        return f"""You are an expert PostgreSQL SQL analyst for an e-commerce clickstream analytics data warehouse.

Your job is to convert natural language questions into accurate, efficient PostgreSQL SELECT queries.

## DATABASE SCHEMA

{self.schema_context}

## BUSINESS CONTEXT

Key tables and their purpose:
- dim_users: User attributes, lifetime metrics, and segments (High Value, Converted, Engaged, New/Inactive)
- dim_products: Product catalog with name, category, price, rating
- dim_product_metrics: Aggregated product performance (clicks, add-to-cart, conversion rates)
- dim_date: Date dimension for time-based analysis
- fact_events: Individual clickstream events (page_view, product_click, add_to_cart, remove_from_cart, checkout, purchase). Grain: one row per event.
- fact_sessions: Session-level aggregates with conversion flag, duration, device info. Grain: one row per session.
- fact_daily_metrics: Daily KPIs (users, sessions, revenue, conversion rate). Grain: one row per day.
- ml_conversion_predictions: ML model predictions for session purchase probability
- ml_churn_predictions: ML model predictions for user churn risk with retention priority score
- ml_ltv_predictions: ML model predictions for 90-day customer lifetime value
- agg_product_daily: Product performance aggregated by day
- agg_user_weekly: User activity aggregated by week
- agg_category_metrics: Category-level performance summary

## RULES

1. Output ONLY a single SQL SELECT statement. No explanations, no markdown, no comments.
2. NEVER use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, or REVOKE.
3. Always use proper table aliases for readability.
4. Use JOINs when data spans multiple tables.
5. Include ORDER BY for ranked/sorted results.
6. Include LIMIT when the user asks for "top N" items.
7. Use appropriate aggregate functions (COUNT, SUM, AVG, etc.) with GROUP BY.
8. Format monetary values with ROUND() to 2 decimal places.
9. Use the pre-built views (vw_*) when they directly answer the question.
10. For conversion-related questions, use fact_sessions.converted column.
11. For revenue questions, use fact_events.cart_value where event_type = 'purchase'.

## EXAMPLES

Question: "What are the top 10 products by revenue?"
SQL: SELECT p.product_id, p.product_name, p.category, ROUND(SUM(e.cart_value)::numeric, 2) AS total_revenue FROM fact_events e JOIN dim_products p ON e.product_id = p.product_id WHERE e.event_type = 'purchase' GROUP BY p.product_id, p.product_name, p.category ORDER BY total_revenue DESC LIMIT 10

Question: "What is the conversion rate by device type?"
SQL: SELECT device, COUNT(*) AS total_sessions, SUM(CASE WHEN converted THEN 1 ELSE 0 END) AS converted_sessions, ROUND(100.0 * SUM(CASE WHEN converted THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_rate FROM fact_sessions GROUP BY device ORDER BY conversion_rate DESC

Question: "How many users are in each segment?"
SQL: SELECT user_segment, COUNT(*) AS num_users, ROUND(SUM(total_revenue)::numeric, 2) AS total_revenue FROM dim_users GROUP BY user_segment ORDER BY num_users DESC

Question: "Show the daily revenue trend"
SQL: SELECT metric_date, unique_users, unique_sessions, total_purchases, ROUND(total_revenue::numeric, 2) AS total_revenue, conversion_rate FROM fact_daily_metrics ORDER BY metric_date

Question: "Which users have the highest churn risk?"
SQL: SELECT u.user_id, u.user_segment, ROUND(c.churn_probability::numeric, 4) AS churn_probability, c.churn_risk, ROUND(c.retention_priority_score::numeric, 2) AS retention_priority FROM ml_churn_predictions c JOIN dim_users u ON c.user_id = u.user_id ORDER BY c.churn_probability DESC LIMIT 20"""

    def generate_sql(self, question: str) -> str:
        """
        Send user question to Ollama and extract the SQL query.

        Args:
            question: Natural language question about the data

        Returns:
            Generated SQL query string

        Raises:
            NLToSQLError: If Ollama API call fails
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": question},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.config["ai"]["temperature"],
                    },
                },
                timeout=120,
            )
            response.raise_for_status()

            result = response.json()
            sql = result["message"]["content"].strip()

            # Strip markdown code fences if present
            sql = re.sub(r"^```(?:sql)?\s*", "", sql)
            sql = re.sub(r"\s*```$", "", sql)
            sql = sql.strip()

            # Remove trailing semicolons
            sql = sql.rstrip(";").strip()

            # If the model returned explanations, try to extract just the SQL
            if not sql.upper().startswith(("SELECT", "WITH")):
                sql_match = re.search(
                    r"((?:SELECT|WITH)\s+.+)",
                    sql,
                    re.IGNORECASE | re.DOTALL,
                )
                if sql_match:
                    sql = sql_match.group(1).rstrip(";").strip()

            return sql

        except requests.ConnectionError:
            raise NLToSQLError(
                "Cannot connect to Ollama. Make sure it's running: ollama serve"
            )
        except requests.Timeout:
            raise NLToSQLError("Ollama request timed out. The model may be loading.")
        except Exception as e:
            raise NLToSQLError(f"Ollama API error: {e}")

    def validate_sql(self, sql: str) -> Tuple[bool, str]:
        """
        Validate that generated SQL is safe to execute.

        Checks:
        1. Query starts with SELECT or WITH
        2. No forbidden keywords (INSERT, DROP, etc.)
        3. No multiple statements

        Args:
            sql: SQL query string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "Empty SQL query"

        # Normalize whitespace
        normalized = " ".join(sql.split())

        # Check first keyword
        first_word = normalized.split()[0].upper()
        if first_word not in self.ALLOWED_START_KEYWORDS:
            return False, f"Query must start with SELECT or WITH, got: {first_word}"

        # Remove string literals to avoid false positives on keywords inside strings
        cleaned = re.sub(r"'[^']*'", "''", normalized)

        # Check for forbidden keywords (as whole words)
        for keyword in self.FORBIDDEN_KEYWORDS:
            pattern = r"\b" + keyword + r"\b"
            if re.search(pattern, cleaned, re.IGNORECASE):
                return False, f"Forbidden SQL keyword detected: {keyword}"

        # Check for multiple statements (semicolon followed by more SQL)
        if ";" in cleaned:
            parts = [p.strip() for p in cleaned.split(";") if p.strip()]
            if len(parts) > 1:
                return False, "Multiple SQL statements not allowed"

        return True, ""

    def execute_query(
        self, sql: str, max_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Execute validated SQL query against PostgreSQL in read-only mode.

        Args:
            sql: Validated SQL query
            max_rows: Maximum rows to return. Defaults to config value.

        Returns:
            Query results as a pandas DataFrame

        Raises:
            QueryExecutionError: If query execution fails
        """
        if max_rows is None:
            max_rows = self.config["ai"]["max_result_rows"]

        conn = None
        try:
            conn = self._get_db_connection()

            # Wrap with LIMIT if not already present
            if "LIMIT" not in sql.upper():
                wrapped_sql = f"SELECT * FROM ({sql}) AS _subq LIMIT {max_rows}"
            else:
                wrapped_sql = sql

            df = pd.read_sql_query(wrapped_sql, conn)
            return df

        except psycopg2.Error as e:
            raise QueryExecutionError(f"SQL execution failed: {e}")
        except Exception as e:
            raise QueryExecutionError(f"Query error: {e}")
        finally:
            if conn:
                conn.close()

    def format_results(self, df: pd.DataFrame, question: str = "") -> str:
        """
        Format query results for terminal display.

        Args:
            df: Query results DataFrame
            question: Original question (for context)

        Returns:
            Formatted string for display
        """
        if df is None or df.empty:
            return "  No results found."

        lines = []

        row_count = len(df)
        col_count = len(df.columns)
        lines.append(f"  Results: {row_count} row(s), {col_count} column(s)")
        lines.append("")

        # Format display
        if row_count <= 50:
            lines.append(df.to_string(index=False))
        else:
            lines.append(df.head(50).to_string(index=False))
            lines.append(f"\n  ... and {row_count - 50} more rows")

        return "\n".join(lines)

    def ask(self, question: str) -> Dict[str, Any]:
        """
        End-to-end pipeline: question -> SQL -> validate -> execute -> format.

        Args:
            question: Natural language question about the data

        Returns:
            Dictionary with keys: question, sql, results_df, formatted_output, error
        """
        result = {
            "question": question,
            "sql": None,
            "results_df": None,
            "formatted_output": None,
            "error": None,
        }

        # Step 1: Generate SQL
        try:
            sql = self.generate_sql(question)
            result["sql"] = sql
        except NLToSQLError as e:
            result["error"] = f"SQL generation failed: {e}"
            return result

        # Step 2: Validate SQL
        is_valid, error_msg = self.validate_sql(sql)
        if not is_valid:
            result["error"] = f"SQL validation failed: {error_msg}"
            return result

        # Step 3: Execute query
        try:
            df = self.execute_query(sql)
            result["results_df"] = df
        except QueryExecutionError as e:
            result["error"] = f"Query execution failed: {e}"
            return result

        # Step 4: Format results
        result["formatted_output"] = self.format_results(df, question)

        return result
