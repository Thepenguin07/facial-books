FacialBooks
FacialBooks is a Python-based facial recognition application designed to manage and identify users through a centralized database system. The project follows a modular architecture, separating the core application logic, database management, and facial recognition engine.

🚀 Features
Database Management: Automated initialization and handling of user records.

Facial Recognition Engine: (Inferred) Processing and identifying faces via a dedicated engine.

Modular Design: Clean separation of concerns between UI/Application logic and back-end services.

🛠 Project Structure
Based on the current codebase, the project is organized as follows:

File	Description
MAINdl.py	The main entry point that initializes the database and launches the app.
Application.py	Contains the FacialBooksApp class; manages the user interface and main loop.
Dbmanager.py	Contains DatabaseManager; handles SQL/NoSQL storage and schema initialization.
Faceengine.py	Houses the facial recognition logic (likely using OpenCV, dlib, or DeepFace).
🔧 Installation & Setup
Clone the repository:

Bash
git clone https://github.com/your-username/FacialBooks.git
cd FacialBooks
Install Dependencies:
(Ensure you have Python 3.8+ installed)

Bash
pip install -r requirements.txt
Run the Application:

Bash
python MAINdl.py
🖥 Usage
Upon execution, the system follows this workflow:

Path Resolution: Automatically adds the project directory to the system path.

Database Sync: Initializes the DatabaseManager to ensure all tables and local storage are ready.

App Launch: Starts the FacialBooksApp instance to begin facial scanning or user management.
