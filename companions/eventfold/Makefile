CC ?= cc
CFLAGS ?= -O3 -std=c99 -Wall -Wextra -Wpedantic
.PHONY: all test clean
all: build/eventfold

build/eventfold: cli/main.c eventfold.h
	mkdir -p build
	$(CC) $(CFLAGS) cli/main.c -o $@

test: all
	PYTHONPATH=src python3 -m unittest discover -s tests -v

clean:
	rm -rf build
