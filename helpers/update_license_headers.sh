#! /bin/sh

for fn in `find helpers MPITools test -name "*.py"`; do
  echo $fn
  python3 helpers/copyright.py $fn | cat - LICENCE.md | python3 helpers/replace_header.py $fn
done