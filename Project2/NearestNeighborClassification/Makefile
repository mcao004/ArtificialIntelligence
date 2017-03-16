#Makefile for Project 2 for CS 170 Artificial Intelligence

CC = g++
CFLAGS = -W -Wall -Werror -ansi -pedantic
H_FILES = src/Instance.h src/dataReader.h src/NNClassifier.h src/LOOValidator.h
CPP_FILES = src/dataReader.cpp src/NNClassifier.cpp src/LOOValidator.cpp src/subsetSearch.cpp
MAIN = src/main.cpp
TEST = src/Test.cpp
OBJECTS = bin/dataReader.o bin/NNClassifier.o bin/LOOValidator.o
FILES = $(MAIN) $(TEST) $(H_FILES) $(CPP_FILES)

all: $(OBJECTS) $(FILES)
	@mkdir -p bin
	$(CC) $(MAIN) $(OBJECTS) src/subsetSearch.cpp $(CFLAGS) -o bin/a.out

test: $(FILES)
	@mkdir -p bin
	$(CC) $(TEST) $(CPP_FILES) $(CFLAGS) -o bin/test.out

bin/dataReader.o: src/dataReader.h src/dataReader.cpp src/Instance.h
	@mkdir -p bin
	$(CC) src/dataReader.cpp $(CFLAGS) -c -o bin/dataReader.o

bin/NNClassifier.o: src/NNClassifier.cpp src/NNClassifier.h src/Instance.h
	@mkdir -p bin
	$(CC) src/NNClassifier.cpp $(CFLAGS) -c -o bin/NNClassifier.o

bin/LOOValidator.o: src/LOOValidator.cpp src/LOOValidator.h src/NNClassifier.h src/Instance.h
	@mkdir -p bin
	$(CC) src/LOOValidator.cpp $(CFLAGS) -c  -o bin/LOOValidator.o

# clean up
clean:
	rm -f bin/*.o bin/*.out
