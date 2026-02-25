# DB Admin Scripts

Administrative utilities for Oracle Vector Search collections.

Run commands from the project root:

```bash
cd /Users/lsaetta/Progetti/multimodal-extraction
```

## 1) Test DB connection

Checks that DB credentials and connectivity are valid.

```bash
python -m scripts.db_admin.test_db_connection
```

## 2) List available collections

Prints all vector collections found in the current Oracle schema.

```bash
python -m scripts.db_admin.list_collections
```

## 3) List documents in a collection

Prints distinct document names from `METADATA.source`.

```bash
python -m scripts.db_admin.list_documents COLL01
```

## 4) Delete documents from a collection

Deletes all chunks for one or more documents, based on `METADATA.source`.

Dry-run example (recommended first):

```bash
python -m scripts.db_admin.delete_documents COLL01 "doc1.pdf" "doc2.pdf" --dry-run
```

Actual delete:

```bash
python -m scripts.db_admin.delete_documents COLL01 "doc1.pdf" "doc2.pdf"
```

## 5) Drop a collection

Drops the whole collection table (destructive action).

```bash
python -m scripts.db_admin.drop_collection COLL01
```

## 6) Optional SQL check

`check_chunks.sql` is available for manual SQL checks in your DB client:

- `scripts/db_admin/check_chunks.sql`
