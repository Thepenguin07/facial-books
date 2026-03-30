"""
Setup.py -- FacialBooks Project Setup & Dependency Installer
Run with: python Setup.py
"""
import subprocess, sys

REQUIRED_PACKAGES = [
    "opencv-python",
    "numpy",
    "Pillow",
    "pandas",
    "openpyxl",
    "customtkinter",
    "tensorflow",        # deep learning framework (replaces face_recognition)
]

def install_packages():
    print("=" * 55)
    print("  FacialBooks -- Dependency Installer")
    print("=" * 55)
    print(f"Python: {sys.version}\n")
    for pkg in REQUIRED_PACKAGES:
        print(f"Installing {pkg}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  OK  {pkg}")
        else:
            print(f"  FAILED  {pkg}")
            print(result.stderr[-300:])
    print("\n" + "=" * 55)
    print("  Setup complete. Create a ./negatives/ folder and add")
    print("  20-50 photos of OTHER people before registering employees.")
    print("  Then run:  python main.py")
    print("=" * 55)

if __name__ == "__main__":
    install_packages()
