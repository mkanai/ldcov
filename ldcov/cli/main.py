"""
Main entry point for the ldcov command-line interface.

This module provides the main entry point for the ldcov command-line interface.
"""

import sys
import logging
from typing import List, Optional

from .commands import run_cli


def main(args: Optional[List[str]] = None):
    """
    Main entry point for the ldcov command-line interface.

    Parameters:
    -----------
    args : list, optional
        Command-line arguments (for testing)
    """
    if args:
        sys.argv = args

    try:
        run_cli()
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
