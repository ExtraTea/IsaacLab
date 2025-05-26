#!/usr/bin/env python
# Test USD import

from pxr import Usd, UsdGeom, UsdUtils

print("Available members in Usd module:")
for member in dir(Usd):
    print(f"  Usd.{member}")
