C = gcc
CXX = g++
CFLAGS = -Wall -O3
ABS2FLAGS = -I../include -L. -labs2c -labs2 -ldl -lstdc++

ALL =  libabs2c.a randomc catc nugc solver_infoc

all: $(ALL)

%: %.c libabs2c.a
	$(C) $< -o $@ $(CFLAGS) $(ABS2FLAGS)

libabs2c.a: libabs2c.o
	ar rcs $@ $<

libabs2c.o: libabs2c.cpp
	$(CXX) -c $< -o $@ $(CFLAGS)

clean:
	/bin/rm -f $(ALL) *.o *.a