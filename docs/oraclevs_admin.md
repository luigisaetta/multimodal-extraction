# `oraclevs_admin.py` — Oracle Vector Store Collection Administration

## What this module provides

This module defines **`OracleVSAdmin`**, an extension of `langchain_oracledb.OracleVS` that adds **administrative / operational utilities** for Oracle Vector Search collections (tables).

It focuses on:
- discovering collections
- listing documents inside a collection (by metadata source)
- basic analysis and maintenance (stats, deletes, drops)

### Public API used by other modules
- **`OracleVSAdmin.list_collections(connection) -> list[str]`**  
  Returns all tables in the current schema that contain at least one `VECTOR` column.

- **`OracleVSAdmin.list_documents_in_collection(connection, collection_name) -> list[str]`**  
  Returns the distinct set of document identifiers found in `METADATA.$.source`.

- **`OracleVSAdmin.list_documents_with_chunk_counts(connection, collection_name) -> list[dict]`**  
  Returns `{"document": <source>, "n_chunks": <count>}` per document, computed with a single `GROUP BY`.
  This is typically used by UIs (e.g., Streamlit inspector) to show what has been loaded.

- **`OracleVSAdmin.analyze_collection(connection, collection_name) -> str`**  
  Produces a short report:
  - total rows (chunks)
  - vector dimensions distribution
  - vector format distribution

- **`OracleVSAdmin.delete_documents(connection, collection_name, doc_names)`**  
  Deletes all rows (chunks) whose `METADATA.$.source` matches one of the provided names.

- **`OracleVSAdmin.drop_collection(connection, collection_name)`**  
  Drops the collection table.

---

## Key design decisions

### 1) Treat `METADATA.$.source` as the document identity
All document-level operations rely on a stable convention:
- each chunk row has a JSON column `METADATA`
- `METADATA.source` identifies the original PDF/document

This enables consistent listing, counting, and deletion without requiring extra columns.

### 2) Identifier safety to avoid SQL injection
Collection names are interpolated into SQL. To keep this safe, the module validates identifiers via:
- `_safe_ident()` (regex whitelist: `^[A-Z][A-Z0-9_$#]*$`)

This is intentionally strict and assumes collection names are standard Oracle identifiers.

### 3) Single-query “inventory” operations
For UI use cases, performance matters. The module prefers:
- a **single `GROUP BY` query** for document counts
instead of:
- many queries per document

This keeps the Streamlit inspector responsive even with large collections.

### 4) Admin methods are read-mostly and explicit about destructive actions
The module separates:
- read operations (list/analyze)
- destructive operations (delete/drop)

Delete/drop are explicit, require a caller to pass the names, and commit changes.

---

## Practical notes / limitations
- The module assumes the table has a column named `METADATA` containing JSON. If you rename columns or use a different schema, queries must be adapted.
- `drop_collection` will permanently remove the table; in a UI, you should gate it with confirmations.
- `analyze_collection` scans all rows; on very large tables it can be slow (acceptable for diagnostics, not for frequent UI refresh).
