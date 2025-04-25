import contextlib
import io
import logging
import sys


@contextlib.contextmanager
def capture_all_output(disable=False):
    if disable:
        # Simply yield control without capturing output.
        yield
        return

    # Prepare in-memory streams.
    new_stdout, new_stderr = io.StringIO(), io.StringIO()
    # Save current streams.
    old_stdout, old_stderr = sys.stdout, sys.stderr
    # Save current logging disable level.
    old_disable = logging.root.manager.disable
    try:
        sys.stdout, sys.stderr = new_stdout, new_stderr
        logging.disable(logging.CRITICAL)  # disable all logging messages below CRITICAL
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        logging.disable(old_disable)  # restore previous logging level
