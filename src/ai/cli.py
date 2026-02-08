"""
Interactive CLI for Natural Language SQL Queries

Provides a REPL interface where users can ask questions about their
e-commerce clickstream data in plain English. Questions are converted
to SQL using the Claude API and executed against PostgreSQL.

Usage:
    python -m src.ai.cli
"""

import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.nl_to_sql import NLToSQLEngine, NLToSQLError


class NLToSQLCLI:
    """
    Interactive CLI for natural language SQL queries.

    Provides a terminal-based REPL that accepts plain English questions,
    converts them to SQL using Claude, and displays the results.
    """

    EXAMPLE_QUERIES: List[str] = [
        "What are the top 10 products by revenue?",
        "How many users are in each segment?",
        "What is the conversion rate by device type?",
        "Show the daily revenue trend for the last 7 days",
        "Which users have the highest churn risk?",
        "What is the average session duration by browser?",
        "Show me the top 5 categories by number of purchases",
    ]

    def __init__(self):
        """Initialize the CLI and the NL-to-SQL engine."""
        self.engine = None

    def display_welcome(self):
        """Print welcome banner with instructions and example queries."""
        print()
        print("=" * 70)
        print("  Natural Language SQL Query Engine")
        print("  Ask questions about your e-commerce data in plain English")
        print("  Powered by Anthropic Claude API")
        print("=" * 70)
        print()
        print("  Example queries you can try:")
        for i, query in enumerate(self.EXAMPLE_QUERIES, 1):
            print(f"    {i}. \"{query}\"")
        print()
        print("  Commands:")
        print("    help      - Show this help message")
        print("    examples  - Show example queries")
        print("    schema    - Show database schema summary")
        print("    quit/exit - Exit the program")
        print()
        print("-" * 70)

    def display_examples(self):
        """Show example queries."""
        print()
        print("  Example queries:")
        for i, query in enumerate(self.EXAMPLE_QUERIES, 1):
            print(f"    {i}. \"{query}\"")
        print()

    def display_schema(self):
        """Show a summary of the database schema."""
        if self.engine and self.engine.schema_context:
            print()
            print("  Database Schema:")
            print("  " + "-" * 40)
            # Show first 50 lines of schema context
            lines = self.engine.schema_context.split("\n")
            for line in lines[:60]:
                print(f"  {line}")
            if len(lines) > 60:
                print(f"  ... and {len(lines) - 60} more lines")
            print()
        else:
            print("  Schema not available.")

    def run(self):
        """
        Main REPL loop.

        Prompts for user input, processes questions through the
        NL-to-SQL engine, and displays results.
        """
        self.display_welcome()

        # Initialize engine
        print("  Initializing engine...")
        try:
            self.engine = NLToSQLEngine()
            print("  Engine ready!")
            print()
        except Exception as e:
            print(f"  ERROR: Failed to initialize engine: {e}")
            print("  Make sure your .env file has ANTHROPIC_API_KEY set")
            print("  and PostgreSQL is running (docker-compose up -d postgres)")
            return

        # REPL loop
        while True:
            try:
                question = input("You> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not question:
                continue

            # Handle commands
            lower_q = question.lower()

            if lower_q in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if lower_q == "help":
                self.display_welcome()
                continue

            if lower_q == "examples":
                self.display_examples()
                continue

            if lower_q == "schema":
                self.display_schema()
                continue

            # Process the question
            print()
            print("  Generating SQL...")

            result = self.engine.ask(question)

            # Display generated SQL
            if result["sql"]:
                print()
                print("  Generated SQL:")
                print("  " + "-" * 50)
                # Indent the SQL for readability
                for line in result["sql"].split("\n"):
                    print(f"    {line}")
                print("  " + "-" * 50)

            # Check for errors
            if result["error"]:
                print()
                print(f"  ERROR: {result['error']}")
                print()
                continue

            # Display results
            print()
            print("  Executing query...")
            print()
            if result["formatted_output"]:
                print(result["formatted_output"])
            print()


def main():
    """Entry point for the NL-to-SQL CLI."""
    cli = NLToSQLCLI()
    cli.run()


if __name__ == "__main__":
    main()
