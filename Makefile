MPICC=mpicc
RUNNER=mpirun
CC=gcc

CFLAGS=-O3

TARGET=jacobi
TARGET-fix=jacobi-fix
HOSTFILE=hostfile

all: $(TARGET)

$(TARGET) : $(TARGET).c
	$(MPICC) $(CFLAGS) -o $@ $<

$(TARGET-fix): $(TARGET-fix).c
	$(CC) $(CFLAGS) -o $@ $<

run-fix: $(TARGET-fix)
	./$(TARGET-fix) 1200

run: $(TARGET)
	$(RUNNER) -n 12 ./$(TARGET) 1200

run-par: $(TARGET)
	$(RUNNER) -n 12 -hostfile $(HOSTFILE) ./$(TARGET)

fullrun: $(TARGET)
	$(RUNNER) -n 12 -hostfile $(HOSTFILE) ./$(TARGET) 67000

clean:
	rm -f $(TARGET) *~ *.o