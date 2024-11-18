#!/bin/sh
# Modify/update requirements.txt for pip and environment.yaml for conda with latest Poetry environment

# For pip, requirements.txt
poetry export --without-hashes -f requirements.txt > tmp
sed -e "s/==.*$//g" tmp > requirements.txt
rm tmp

# For conda, envirnment.yaml
poetry2conda pyproject.toml environment.yaml
