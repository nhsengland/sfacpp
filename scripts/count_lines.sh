#! /bin/bash

cloc . \
    --by-file-by-lang \
    --fullpath \
    --not-match-d="(modules|.venv|build)" \
    --not-match-f="((include/math/primes.hpp)|.*.json|.*.yml|.*.yaml|.*.txt|.*.html|.*.svg|.*.tex|.*.csv|.*.parquet|.*.scss|.*.css)" \
    --report-file code_count.txt