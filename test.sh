#!/bin/sh
set -e

for f in test/**/*.py; do
  echo 🍜 Testing "$f"...
  python "$f"
  echo
done
