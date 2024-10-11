import os
import time
from typing import List, Dict
import pandas as pd
from neo4j import GraphDatabase

# Database configuration
DB_CONFIG = {
    "url": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "password12345",
    "database": "neo4j",
    "index_name": "entity"
}


driver = GraphDatabase.driver(DB_CONFIG["url"], auth=(DB_CONFIG["username"], DB_CONFIG["password"]))

def batched_import(statement: str, df: pd.DataFrame, batch_size: int = 1000) -> int:
    """
    Import a dataframe into Neo4j using a batched approach.

    Args:
        statement (str): The Cypher query to execute.
        df (pd.DataFrame): The dataframe to import.
        batch_size (int): The number of rows to import in each batch.

    Returns:
        int: Total number of rows imported.
    """
    total = len(df)
    start_time = time.time()
    for start in range(0, total, batch_size):
        batch = df.iloc[start:min(start + batch_size, total)]
        result = driver.execute_query(
            "UNWIND $rows AS value " + statement,
            rows=batch.to_dict('records'),
            database_=DB_CONFIG["database"]
        )
        print(result.summary.counters)
    print(f'{total} rows imported in {time.time() - start_time:.2f} seconds.')
    return total

def create_constraints():
    """Create necessary constraints in the database."""
    statements = [
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:__Chunk__) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:__Document__) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (c:__Community__) REQUIRE c.community IS UNIQUE",
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:__Entity__) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT entity_title IF NOT EXISTS FOR (e:__Entity__) REQUIRE e.name IS UNIQUE",
        "CREATE CONSTRAINT entity_title IF NOT EXISTS FOR (e:__Covariate__) REQUIRE e.title IS UNIQUE",
        "CREATE CONSTRAINT related_id IF NOT EXISTS FOR ()-[rel:RELATED]->() REQUIRE rel.id IS UNIQUE"
    ]

    for statement in statements:
        print(f"Executing: {statement}")
        driver.execute_query(statement)

def import_documents(graph_folder: str):
    """Import documents into the database."""
    doc_df = pd.read_parquet(f'{graph_folder}/output/create_final_documents.parquet', columns=["id", "title"])
    statement = """
    MERGE (d:__Document__ {id: value.id})
    SET d += value {.title}
    """
    batched_import(statement, doc_df)

def import_text_units(graph_folder: str):
    """Import text units into the database."""
    text_df = pd.read_parquet(f'{graph_folder}/output/create_final_text_units.parquet',
                              columns=["id", "text", "n_tokens", "document_ids"])
    statement = """
    MERGE (c:__Chunk__ {id: value.id})
    SET c += value {.text, .n_tokens}
    WITH c, value
    UNWIND value.document_ids AS document
    MATCH (d:__Document__ {id: document})
    MERGE (c)-[:PART_OF]->(d)
    """
    batched_import(statement, text_df)

def import_entities(graph_folder: str):
    """Import entities into the database."""
    entity_df = pd.read_parquet(f'{graph_folder}/output/create_final_entities.parquet',
                                columns=["name", "type", "description", "human_readable_id", "id", "description_embedding", "text_unit_ids"])
    entity_statement = """
    MERGE (e:__Entity__ {id: value.id})
    SET e += value {.human_readable_id, .description, name: replace(value.name, '"', ''), description_embedding: value.description_embedding}
    WITH e, value
    CALL apoc.create.addLabels(e, CASE WHEN coalesce(value.type, "") = "" THEN [] ELSE [apoc.text.upperCamelCase(replace(value.type, '"', ''))] END) YIELD node
    UNWIND value.text_unit_ids AS text_unit
    MATCH (c:__Chunk__ {id: text_unit})
    MERGE (c)-[:HAS_ENTITY]->(e)
    """
    batched_import(entity_statement, entity_df)

def import_relationships(graph_folder: str):
    """Import relationships into the database."""
    rel_df = pd.read_parquet(f'{graph_folder}/output/create_final_relationships.parquet',
                             columns=["source", "target", "id", "rank", "weight", "human_readable_id", "description", "text_unit_ids"])
    rel_statement = """
    MATCH (source:__Entity__ {name: replace(value.source, '"', '')})
    MATCH (target:__Entity__ {name: replace(value.target, '"', '')})
    MERGE (source)-[rel:RELATED {id: value.id}]->(target)
    SET rel += value {.rank, .weight, .human_readable_id, .description, .text_unit_ids}
    RETURN count(*) AS createdRels
    """
    batched_import(rel_statement, rel_df)

def import_communities(graph_folder: str):
    """Import communities into the database."""
    community_df = pd.read_parquet(f'{graph_folder}/output/create_final_communities.parquet',
                                   columns=["id", "level", "title", "text_unit_ids", "relationship_ids"])
    statement = """
    MERGE (c:__Community__ {community: value.id})
    SET c += value {.level, .title}
    WITH *
    UNWIND value.relationship_ids AS rel_id
    MATCH (start:__Entity__)-[:RELATED {id: rel_id}]->(end:__Entity__)
    MERGE (start)-[:IN_COMMUNITY]->(c)
    MERGE (end)-[:IN_COMMUNITY]->(c)
    RETURN count(DISTINCT c) AS createdCommunities
    """
    batched_import(statement, community_df)

def import_community_reports(graph_folder: str):
    """Import community reports into the database."""
    community_report_df = pd.read_parquet(f'{graph_folder}/output/create_final_community_reports.parquet',
                                          columns=["id", "community", "level", "title", "summary", "findings", "rank", "rank_explanation", "full_content"])
    community_statement = """
    MERGE (c:__Community__ {community: value.community})
    SET c += value {.level, .title, .rank, .rank_explanation, .full_content, .summary}
    WITH c, value
    UNWIND range(0, size(value.findings)-1) AS finding_idx
    WITH c, value, finding_idx, value.findings[finding_idx] AS finding
    MERGE (c)-[:HAS_FINDING]->(f:Finding {id: finding_idx})
    SET f += finding
    """
    batched_import(community_statement, community_report_df)

def import_microsoft_graph(graph_folder: str):
    """Main function to orchestrate the import process."""
    create_constraints()
    import_documents(graph_folder)
    import_text_units(graph_folder)
    import_entities(graph_folder)
    import_relationships(graph_folder)
    import_communities(graph_folder)
    import_community_reports(graph_folder)