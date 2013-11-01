CC=mpicc

TARGET=jacobi

all: $(TARGET)

% : %.c
	$(CC) -o $@ $<

clean:
	rm -f $(TARGET) *~ *.o