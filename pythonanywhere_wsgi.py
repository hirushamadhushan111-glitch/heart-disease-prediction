"""
WSGI config for hosting the Flask app on PythonAnywhere (free tier).

This file is NOT used when running locally. To deploy:

  1. Bash console:
         git clone https://github.com/hirushamadhushan111-glitch/heart-disease-prediction.git
         cd heart-disease-prediction
         python3 train_model.py --quick

     --quick rebuilds heart_model.pkl using whatever scikit-learn version
     PythonAnywhere has installed. Without it the pickle was built by a
     different version and may refuse to load.

  2. Web tab -> Add a new web app -> Manual configuration -> Python 3.x

  3. Open the "WSGI configuration file" link on the Web tab, delete
     everything in it, and paste the contents of this file.

  4. Change USERNAME below to your PythonAnywhere username.

  5. Hit the green Reload button.

app.py resolves its own paths relative to __file__, so there is no need to
change the working directory here.
"""

import sys
from pathlib import Path

USERNAME = 'YOUR_USERNAME'          # <-- change this

APP_DIR = Path('/home') / USERNAME / 'heart-disease-prediction' / 'heart-disease-app'

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# PythonAnywhere looks for a module-level name called `application`
from app import app as application   # noqa: E402
