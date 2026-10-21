"""
studybuddy/scripts/reset_admin_password.py

Sets a new password for an existing admin user. Run it once if the admin account was
created with the default password that used to be written in create_admin.py: that
password is still visible in this repository's history, so it must not stay in use.

Run from the project root (Studybuddy/):
    python studybuddy/scripts/reset_admin_password.py              # the 'admin' user
    python studybuddy/scripts/reset_admin_password.py other_admin  # another admin

It connects with MONGO_URI from studybuddy/.env and asks for the new password twice.
"""

import getpass
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent   # studybuddy/scripts/
PKG_DIR    = SCRIPT_DIR.parent                 # studybuddy/

sys.path.insert(0, str(PKG_DIR))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import bcrypt

from src.db.mongo_client import get_users_collection

MIN_ADMIN_PASSWORD_LEN = 12


def main() -> int:
    """Replace the admin's bcrypt hash with one for a new password. Returns an exit code."""
    username = (sys.argv[1] if len(sys.argv) > 1 else "admin").strip().lower()
    users = get_users_collection()
    user = users.find_one({"username": username}, {"role": 1})
    if not user:
        print(f"❌  No user named '{username}'. Nothing changed.")
        return 1
    if user.get("role") != "admin":
        print(f"❌  '{username}' isn't an admin; this script only resets admin passwords.")
        return 1

    password = getpass.getpass("New admin password: ")
    if password != getpass.getpass("Type it again: "):
        print("❌  The two passwords don't match. Nothing changed.")
        return 1
    if len(password) < MIN_ADMIN_PASSWORD_LEN:
        print(f"❌  Use at least {MIN_ADMIN_PASSWORD_LEN} characters. Nothing changed.")
        return 1

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    result = users.update_one({"username": username}, {"$set": {"password_hash": password_hash}})
    if result.matched_count != 1:
        print("❌  The password wasn't updated. Check MONGO_URI in studybuddy/.env and retry.")
        return 1
    print(f"✅  New password set for '{username}'. The old one no longer works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
