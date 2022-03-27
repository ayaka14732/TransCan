#!/bin/sh

for f in test_*.py; do
  echo Testing "$f"...
  python "$f"
done
