import os
import warnings

# Suppress all warnings and TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from app import app

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)