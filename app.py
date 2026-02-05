import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from ui.streamlit_app import main

if __name__ == "__main__":
    main()
