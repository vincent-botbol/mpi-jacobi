CC=mpicc
RUNNER=mpirun

CFLAGS=-O3

TARGET=jacobi
HOSTFILE=hostfile

all: $(TARGET)

$(TARGET) : $(TARGET).c
	$(CC) $(CFLAGS) -o $@ $<

run: $(TARGET)
	$(RUNNER) -n 12 -hostfile $(HOSTFILE) ./$(TARGET)

fullrun: $(TARGET)
	$(RUNNER) -n 12 -hostfile $(HOSTFILE) ./$(TARGET) 67000

clean:
	rm -f $(TARGET) *~ *.o