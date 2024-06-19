#!/bin/bash

#This script updates the model config dvc files of model and data and commits everything to git
#please git commit and git tag if you want to version this model

dvc add model.pkl
dvc commit model.pkl
git add --all
