#! /bin/sh
# Updates all Python files with license taken from README.md and copyright information obtained from the git log.

for fn in `find examples helpers NuMPI test -name "*.py"`; do
  echo $fn
  python3 maintenance/copyright.py $fn | cat - LICENCE.md | python3 maintenance/replace_header.py $fn
done