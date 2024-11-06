#!/bin/bash
python -m black .

arr=("models" "src")
for elem in "${arr[@]}"
do
  darglint -s sphinx "${elem}/."
  pyflakes "${elem}/."
  isort --profile black "${elem}/."
done