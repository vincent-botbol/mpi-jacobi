CC=mpicc
CFLAGS=-O3

TARGET=jacobi

all: $(TARGET)

$(TARGET) : $(TARGET).c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET) *~ *.o