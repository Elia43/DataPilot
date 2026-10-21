"""
studybuddy/scripts/create_admin.py

Creates the admin user (username: admin). The password is never written in the code:
it is read from ADMIN_PASSWORD in studybuddy/.env, or asked for at the prompt.

Run from the project root (Studybuddy/):
    python studybuddy/scripts/create_admin.py

To change the password of an admin that already exists, run reset_admin_password.py.
"""

import getpass
import os
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
MIN_ADMIN_PASSWORD_LEN = 12


def read_password() -> str:
    """Return the admin password from ADMIN_PASSWORD, or ask for it twice at the prompt."""
    password = os.environ.get("ADMIN_PASSWORD", "")
    if not password:
        password = getpass.getpass("New admin password: ")
        if password != getpass.getpass("Type it again: "):
            raise ValueError("The two passwords don't match.")
    if len(password) < MIN_ADMIN_PASSWORD_LEN:
        raise ValueError(f"Use at least {MIN_ADMIN_PASSWORD_LEN} characters for the admin password.")
    return password


def main():
    print(f"\nCreating admin user: '{ADMIN_USERNAME}'…")
    try:
        user = create_admin_user(ADMIN_USERNAME, read_password())
        print(f"✅  Admin user created successfully.")
        print(f"    Username : {user['username']}")
        print(f"    Role     : admin")
        print(f"\nYou can now log in at the DataPilot login screen.")
    except ValueError as e:
        print(f"⚠️   {e}")
        print("    If the admin user already exists, use reset_admin_password.py to change its password.")
    except Exception as e:
        print(f"❌  Unexpected error: {e}")
        print("    Check that MONGO_URI is set in studybuddy/.env and retry.")


if __name__ == "__main__":
    main()
