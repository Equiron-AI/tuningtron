#!/bin/bash

rm -rf ./dist
python -m build
twine upload dist/*
rm -rf ./dist
rm -rf ./src/tuningtron.egg-info
rm -rf ./src/tuningtron/__pycache__