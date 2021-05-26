#!/bin/bash

echo "running clang format. pwd: $(pwd)"

for file in $(find src -type f)
do
  diff ${file} <(clang-format ${file}) >> ${file}.patch
  echo ${file}.patch
  wc -l ${file}.patch
done
