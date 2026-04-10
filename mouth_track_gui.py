#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper — launches the mouth_track_gui package.

This file exists so that ``python mouth_track_gui.py`` and
``mouth_track_gui.bat`` continue to work after the package migration.
"""
from mouth_track_gui.app import main

if __name__ == "__main__":
    main()
