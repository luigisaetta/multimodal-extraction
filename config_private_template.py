"""
Author: Luigi Saetta
Date last modified: 2024-04-28
Python Version: 3.11

    Template for building config_private.py file
    Put you private configuration settings here
License: MIT
"""

# to test ADB
VECTOR_DB_USER = "your_db_username"
VECTOR_DB_PWD = "your_pwd"
VECTOR_DSN = "your_dsn"

# needed to connect to ADB
VECTOR_WALLET_DIR = "/wallet_atp"
VECTOR_WALLET_PWD = "my_password"

CONNECT_ARGS = {
    "user": VECTOR_DB_USER,
    "password": VECTOR_DB_PWD,
    "dsn": VECTOR_DSN,
    "config_dir": VECTOR_WALLET_DIR,
    "wallet_location": VECTOR_WALLET_DIR,
    "wallet_password": VECTOR_WALLET_PWD,
}

# oraemeaint
COMPARTMENT_ID = "your.ocid"
