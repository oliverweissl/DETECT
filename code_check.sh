#!/bin/bash
python -m black . --line-length 100 --preview

arr=("models" "src" "examples")
for elem in "${arr[@]}"
do
  darglint -s sphinx "${elem}/."
  pyflakes "${elem}/."
  isort --profile black "${elem}/."
done