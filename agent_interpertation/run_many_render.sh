#!/bin/bash
source ../../venv/bin/activate
for i in $(seq 1 "$1")
do
  python draw_world.py "$i" "$2" &
done
