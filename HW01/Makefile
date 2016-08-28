LIBOPTS = -shared
CFLAGS = -fPIC -std=gnu99 -fopenmp
CC = gcc

libconv.so : conv.c
	$(CC) $< $(LIBOPTS) $(CFLAGS) -o $@

clean :
	rm *.so