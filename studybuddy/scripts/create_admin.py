"""
studybuddy/scripts/create_admin.py

Creates the default admin user (username: admin, password: admin1234).

Run from the project root (Studybuddy/):
    python studybuddy/scripts/create_admin.py
"""

import sys
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent   # studybuddy/scripts/
PKG_DIR      = SCRIPT_DIR.parent                 # studybuddy/
PROJECT_ROOT = PKG_DIR.parent                    # Studybuddy/

sys.path.insert(0, str(PKG_DIR))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src.db.mongo_client import create_admin_user

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin1234"


def main():
    print(f"\nCreating admin user: '{ADMIN_USERNAME}'…")
    try:
        user = create_admin_user(ADMIN_USERNAME, ADMIN_PASSWORD)
        print(f"✅  Admin user created successfully.")
        print(f"    Username : {user['username']}")
        print(f"    Role     : admin")
        print(f"\nYou can now log in at the DataPilot login screen.")
    except ValueError as e:
        print(f"⚠️   {e}")
        print("    If the admin user already exists, you can log in directly.")
    except Exception as e:
        print(f"❌  Unexpected error: {e}")
        print("    Check that MONGO_URI is set in studybuddy/.env and retry.")


if __name__ == "__main__":
    main()
