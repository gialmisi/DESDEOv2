#!/bin/bash
source check.sh

black --line-length $LINE_LENGTH "$@"

