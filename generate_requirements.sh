#!/bin/bash
source check.sh
OUTPUT='./requirements.txt'
OUTPUT_DEV='./requirements-dev.txt'
# Base requirements used for developing
DEV_BASE_REQS='./requirements-dev-base.txt'

printf "${GREEN}Generating requirements file for the user${NC}\n"

# Generate dependencies for the user
pipreqs --use-local --print $MODULE_DIR | sort | uniq -u > $OUTPUT

# Generate dependencies for the developer (append)
# Append the base dev requirements
cat $DEV_BASE_REQS > $OUTPUT_DEV
pipreqs --use-local --print $TEST_DIR | sort | uniq -u >> $OUTPUT_DEV
cat $OUTPUT >> $OUTPUT_DEV
# Remove duplicates
tmp=$(mktemp)
cat $OUTPUT_DEV | sort | uniq > $tmp
cat $tmp > $OUTPUT_DEV

printf "${GREEN}All done!${NC}\n"

