#!/bin/bash
source check.sh

ALL_GOOD=true

# Format the code using black
printf "${GREEN}Formatting code with black...${NC}\n"
./black.sh $MODULE_DIR
check_exit_status $? $MODULE_DIR
./black.sh $TEST_DIR
check_exit_status $? $TEST_DIR

# Format the imports using isort
printf "${GREEN}Formatting imports with isort...${NC}\n"
isort --recursive $MODULE_DIR
check_exit_status $? $MODULE_DIR
isort --recursive $TEST_DIR
check_exit_status $? $TEST_DIR

if $ALL_GOOD; then
    # Formatting successfull, run checks
    printf "${GREEN}Formatting done. Runnings final checks...${NC}\n"
    ./check.sh
else
    printf "${RED}Formatting unsuccessful! Check the ouput of black and isort!${NC}\n"
fi
