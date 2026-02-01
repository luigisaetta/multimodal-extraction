# `db_utils.py` — Oracle DB Connection & Small Introspection Helpers

## What this module provides

This module centralizes **Oracle DB connectivity** and a few small helper utilities used by the Streamlit app (and any other modules that need DB access).

### Public API used by other modules

- **`get_db_connection()`**
  - Returns an `oracledb.Connection` using connection parameters defined in `config_private.CONNECT_ARGS`.
  - Logs the connection attempt (user + DSN).
  - Raises on connection failure (does not swallow errors).

- **`get_connection_params() -> dict[str, Any]`**
  - Returns a UI-friendly dictionary of connection parameters:
    - user, dsn, wallet dir
    - password is **masked** via `utils.mask_secret()`
    - includes `COLLECTION_NAME` from `config`
  - Used by the Streamlit “DB / Collection Inspector” page to display config safely.

- **`check_db_connection() -> tuple[bool, str]`**
  - Connectivity test: opens a connection and runs `SELECT 1 FROM dual`.
  - Returns `(True, "Connection OK.")` on success.
  - Returns `(False, "<ExceptionType>: <msg>")` on failure.
  - Designed for UI usage (no exception propagation).

- **`get_table_comment(conn, table_name: str) -> str`**
  - Reads the table comment from `USER_TAB_COMMENTS`.
  - Returns a single string (or empty if missing).
  - Useful for lightweight schema introspection / debugging.

- **`print_table_comment(conn, table_name: str) -> None`**
  - Convenience wrapper that prints the table comment to stdout.

---

## Key design decisions

### 1) Single source of truth for connection arguments
Connection details are not reassembled ad-hoc in multiple modules.
Instead:
- `config_private.CONNECT_ARGS` is the canonical connection definition.
- This reduces divergence and accidental mismatches (dsn/wallet/user/etc.).

### 2) Mask secrets at the boundary
`get_connection_params()` is intentionally separated from raw config usage:
- it masks `VECTOR_DB_PWD` before returning values for UI/logging.
- avoids accidental leaks in Streamlit or logs.

### 3) Two levels of “connection access”
- `get_db_connection()` is the low-level primitive (raises on failure).
- `check_db_connection()` is the safe UI-friendly probe (returns status + message).

This avoids mixing UI concerns (friendly messages) with core DB access.

### 4) Introspection kept minimal and read-only
Table comment helpers are intentionally small and non-invasive:
- they use Oracle data dictionary views (`USER_TAB_COMMENTS`)
- they are safe defaults for diagnostics without requiring extra privileges.

---

## Practical notes
- The module assumes `oracledb` is installed and that `CONNECT_ARGS` includes any required wallet/TLS settings for your environment.
- Logging currently prints user and dsn (useful for debugging); if you consider DSN sensitive in your context, you may want to partially mask it as well.
