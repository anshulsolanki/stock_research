#!/bin/bash
# Helper script to run the Stock Research App using the correct virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run app using the local virtual environment
echo "Starting app using virtual environment..."
"$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/website_ui/app.py"
