#!/bin/sh
set -e

for f in test/**/*.py; do
  echo 🍜 Testing "$f"...
  ./startpod python "$f"  # TODO: ensure that tests can be run on single hosts
  echo
done
