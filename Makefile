MPICC=mpicc
RUNNER=mpirun
CC=gcc

CFLAGS=-O3

TARGET=jacobi
TARGET-fix=jacobi-fix
HOSTFILE=hostfile-sar

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
	rm -f $(TARGET-fix) $(TARGET) *~ *.o

#for i in 55000 56000 57000 58000 59000 60000 61000 62000 63000 64000 65000 66000 67000; do mpirun -n 12 -hostfile hostfile-sar ./jacobi $i; done
