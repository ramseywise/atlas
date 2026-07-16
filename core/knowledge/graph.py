"""
Neo4j knowledge graph for financial entities and relationships.

Node types:   Customer, Segment, Merchant, Industry, MetricDefinition
Edge types:   BELONGS_TO, TRANSACTS_WITH, OPERATES_IN, DEFINES

Used by the knowledge agent to answer:
  - "What does burn ratio mean?"
  - "Which customers are in my segment?"
  - "What are the top merchants for segment X?"
  - "What drives runway for high-growth SaaS companies?"
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from neo4j import GraphDatabase

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")


class AtlasGraph:
    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    @contextmanager
    def session(self):
        with self._driver.session() as s:
            yield s

    # ── Schema bootstrap ──────────────────────────────────────────────────────

    def bootstrap_schema(self):
        """Create constraints and indexes on first run."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Customer) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Segment) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Merchant) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:MetricDefinition) REQUIRE d.name IS UNIQUE",
        ]
        with self.session() as s:
            for c in constraints:
                s.run(c)

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert_customer(self, customer_id: str, props: dict[str, Any]):
        with self.session() as s:
            s.run(
                "MERGE (c:Customer {id: $id}) SET c += $props",
                id=customer_id,
                props=props,
            )

    def upsert_segment(self, segment_id: int, label: str, description: str):
        with self.session() as s:
            s.run(
                "MERGE (s:Segment {id: $id}) SET s.label = $label, s.description = $desc",
                id=segment_id,
                label=label,
                desc=description,
            )

    def assign_customer_to_segment(self, customer_id: str, segment_id: int):
        with self.session() as s:
            s.run(
                """
                MATCH (c:Customer {id: $cid})
                MATCH (s:Segment {id: $sid})
                MERGE (c)-[:BELONGS_TO]->(s)
                """,
                cid=customer_id,
                sid=segment_id,
            )

    def upsert_metric_definition(self, name: str, definition: str, formula: str | None = None):
        with self.session() as s:
            s.run(
                "MERGE (d:MetricDefinition {name: $name}) SET d.definition = $def, d.formula = $formula",
                name=name,
                definition=definition,
                formula=formula,
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_segment_customers(self, segment_id: int) -> list[str]:
        with self.session() as s:
            result = s.run(
                "MATCH (c:Customer)-[:BELONGS_TO]->(s:Segment {id: $sid}) RETURN c.id",
                sid=segment_id,
            )
            return [r["c.id"] for r in result]

    def get_customer_segment(self, customer_id: str) -> dict | None:
        with self.session() as s:
            result = s.run(
                """
                MATCH (c:Customer {id: $cid})-[:BELONGS_TO]->(s:Segment)
                RETURN s.id AS id, s.label AS label, s.description AS description
                """,
                cid=customer_id,
            )
            row = result.single()
            return dict(row) if row else None

    def lookup_metric(self, name: str) -> dict | None:
        with self.session() as s:
            result = s.run(
                "MATCH (d:MetricDefinition {name: $name}) RETURN d.definition AS definition, d.formula AS formula",
                name=name,
            )
            row = result.single()
            return dict(row) if row else None

    def search_metrics(self, keyword: str) -> list[dict]:
        with self.session() as s:
            result = s.run(
                """
                MATCH (d:MetricDefinition)
                WHERE toLower(d.name) CONTAINS toLower($kw)
                   OR toLower(d.definition) CONTAINS toLower($kw)
                RETURN d.name AS name, d.definition AS definition
                LIMIT 5
                """,
                kw=keyword,
            )
            return [dict(r) for r in result]
