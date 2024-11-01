"""Automated tilt-series alignment for cryo-ET."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tttsa")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Marten Chaillet"
__email__ = "martenchaillet@gmail.com"

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from .main import tilt_series_alignment
