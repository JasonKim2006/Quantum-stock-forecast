"""Start the Streamlit dashboard:  python run_dashboard.py

Why not just ``streamlit run app.py``? Streamlit executes the script in a new
worker thread on every interaction. If Qiskit's compiled (Rust) core is first
imported inside one of those threads, using it from the next one can crash
the whole server with a segmentation fault (seen with qiskit 2.6 on Python
3.13). Importing Qiskit here, in the main thread, before Streamlit starts
avoids that.
"""

import os
import sys
from pathlib import Path

import qiskit  # noqa: F401  -- must be imported in the main thread, see above
import qiskit_algorithms  # noqa: F401
from qiskit import QuantumCircuit

QuantumCircuit(1)  # force the Rust extension to initialise here

from streamlit.web import cli  # noqa: E402

if __name__ == "__main__":
    app = Path(__file__).with_name("app.py")
    os.chdir(app.parent)  # so .streamlit/config.toml (theme) is found
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    sys.exit(cli.main())
