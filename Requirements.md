🐍 Install Python on Windows via PowerShell
✅ Method 1 (Recommended): Using winget

Fastest and cleanest.

Open PowerShell as Administrator

Run:

winget install Python.Python.3


This installs the latest stable Python 3 and automatically sets PATH.

🔍 Verify Installation

Close and reopen PowerShell, then run:

python --version
pip --version


You should see Python 3.x and pip versions.

🧠 If python doesn’t work (Windows alias issue)

Disable Microsoft Store alias:

Open Settings

Go to Apps → Advanced app settings → App execution aliases

Turn OFF:

python.exe

python3.exe

Restart PowerShell and try again.

🐍 Method 2: Manual Installer (Official Python)

If you prefer GUI or winget isn’t available:

Download from:
👉 https://www.python.org/downloads/windows/

Run installer

✅ Check “Add Python to PATH”

Click Install Now

Then verify:

python --version

🧪 (Optional) Create a Virtual Environment

Highly recommended for projects:

python -m venv venv
venv\Scripts\activate


You’ll see (venv) in your prompt.

📦 Install Common Packages

Once Python is ready:

pip install pandas numpy matplotlib seaborn scipy scikit-learn

🧯 Troubleshooting Quick Fixes

PowerShell can’t find python?

where python


Update pip:

python -m pip install --upgrade pip
