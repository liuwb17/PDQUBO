CXX = g++
CFLAGS = -Wall -std=c++17 -O3
ABS2FLAGS = -I../include -L../lib -labs2 -ldl

ALL = random cat nqueens nug solver_info

all: $(ALL)

%: %.cpp
	$(CXX) $< -o $@ $(CFLAGS) $(ABS2FLAGS)

clean:
	/bin/rm -f $(ALL)

