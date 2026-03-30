import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Dbmanager import DatabaseManager
from Application import FacialBooksApp

def main():
    print("=" * 50)
    print("  FacialBooks - Starting Application...")
    print("=" * 50)
    db = DatabaseManager()
    db.initialize()
    print("[OK] Database initialized.")
    app = FacialBooksApp(db)
    app.run()

if __name__ == "__main__":
    main()
