#!/usr/bin/env python3
"""Compatibility wrapper for the catalog-backed ERA5 downloader.

Use this command for all new downloads. Legacy `dl_*` scripts remain untouched.
"""

from mwr_retrieval.downloads.era5 import main

if __name__ == "__main__":
    raise SystemExit(main())
