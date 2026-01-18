"""
Database reset utility
Use this to recreate the database with the new schema
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager


def reset_database():
    """Delete and recreate the database"""
    db_path = 'dsa_tutor.db'
    
    # Backup old database if it exists
    if os.path.exists(db_path):
        backup_path = f'{db_path}.backup'
        print(f"[WARNING] Backing up existing database to {backup_path}")
        
        # Remove old backup if exists
        if os.path.exists(backup_path):
            os.remove(backup_path)
        
        # Rename current to backup
        os.rename(db_path, backup_path)
        print(f"[SUCCESS] Backup created")
    
    # Create new database with updated schema
    print(f"[INFO] Creating new database with authentication schema...")
    db = DatabaseManager(db_path)
    print(f"[SUCCESS] Database created successfully!")
    
    # Create a test user
    from utils.auth import hash_password
    
    test_username = "testuser"
    test_email = "test@example.com"
    test_password = "TestPass123"
    
    hashed_password = hash_password(test_password)
    
    if db.register_user(test_username, test_email, hashed_password):
        print(f"\n[SUCCESS] Test user created:")
        print(f"   Username: {test_username}")
        print(f"   Email: {test_email}")
        print(f"   Password: {test_password}")
        print(f"\n   You can use these credentials to login!")
    else:
        print(f"\n[WARNING] Could not create test user (may already exist)")
    
    print(f"\n[SUCCESS] Database reset complete!")
    print(f"\n[NOTE] Old database backed up to {db_path}.backup")


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("DSA Tutor - Database Reset Utility")
    print("=" * 60)
    print()
    
    # Check if --force flag is provided
    if "--force" in sys.argv or "-f" in sys.argv:
        reset_database()
    else:
        print("WARNING: This will DELETE the current database.")
        print("Run with --force or -f flag to proceed:")
        print("  python reset_database.py --force")

