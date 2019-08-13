#!/bin/bash
RED='\033[0;31m'
BRED='\033[0;31;5m'
GREEN='\033[0;32m'
BBLUE='\033[0;34;5m'
NC='\033[0m'

MODULE_DIR='./desdeo/'
EXAMPLE_DIR'./notebooks/'
TEST_DIR='./tests/'
LINE_LENGTH=79

ALL_GOOD=true

check_exit_status () {
    [ $1 -eq 0 ] && : || \
	    { printf "${RED}$2 has some issues${NC}\n"; ALL_GOOD=false; }
}

main () {
    # Check formatting using black
    printf "${GREEN}Checking formatting with black...${NC}\n"
    ./black.sh --check $MODULE_DIR &> /dev/null
    check_exit_status $? $MODULE_DIR
    ./black.sh --check $TEST_DIR &> /dev/null
    check_exit_status $? $TEST_DIR

    # Check that the code conforms with PEPs
    printf "${GREEN}Checking PEPs with flake8...${NC}\n"
    flake8 $MODULE_DIR > /dev/null 
    check_exit_status $? $MODULE_DIR
    flake8 $TEST_DIR > /dev/null 
    check_exit_status $? $TEST_DIR

    # Check that typehints match using mypy
    printf "${GREEN}Checking type hints with mypy...${NC}\n"
    mypy $MODULE_DIR > /dev/null 
    check_exit_status $? $MODULE_DIR

    # Check that the imports are sorted correctly using isort
    printf "${GREEN}Checking imports with isort...${NC}\n"
    isort --recursive --check-only $MODULE_DIR > /dev/null
    check_exit_status $? $MODULE_DIR
    isort --recursive --check-only $TEST_DIR > /dev/null
    check_exit_status $? $TEST_DIR

    # Check tests with pytest
    printf "${GREEN}Running all tests with pytest...\n${NC}"
    pytest --quiet --reruns 5 > /dev/null
    [ $? -eq 0 ] && : || \
	    { printf "${BRED}CRITICAL! ${RED}NOT ALL TESTS PASS!!!${NC}\n"; \
	      ALL_GOOD=false; }

    if $ALL_GOOD; then
	printf "${GREEN}\U1F389 All checks passed! "
	printf "Feel free to ${BBLUE}commit and push${NC}! \U1F389 ${NC}\n"
    else
	printf "${RED}\U1F525 Some issue(s) encountered.\
	Try running format.sh! \U1F525 ${NC}\n"
    fi
}

# Run main if executed, useful when this file is sourced somewhere else
[[ $0 == "./check.sh" ]] && main || :
