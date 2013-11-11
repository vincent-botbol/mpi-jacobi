MPICC=mpicc
RUNNER=mpirun
CC=gcc

CFLAGS=-O3 -fopenmp -Wall

TARGET=jacobi
TARGET-fix=jacobi-fix
HOSTFILE=hostfile-sar

PDFCC=pdflatex
PDFTARGET=rapport_PPAR_Botbol_Bismuth.pdf
PDFSRC=rapport_PPAR_Botbol_Bismuth.tex

all: $(TARGET) $(PDFTARGET)

exec: $(TARGET)
	$(RUNNER) -n 12 -hostfile $(HOSTFILE) ./$(TARGET) 67000

clean:
	rm -f $(TARGET-fix) $(TARGET) *~ *.o *.out *.snm *.log *.aux *.nav *.toc $(PDFTARGET)

$(TARGET) : $(TARGET).c
	$(MPICC) $(CFLAGS) -o $@ $<

omp: $(TARGET).c
	$(MPICC) $(CFLAGS) -o $(TARGET) $< -DOMP_STRAT

$(TARGET-fix): $(TARGET-fix).c
	$(CC) $(CFLAGS) -o $@ $<

run-fix: $(TARGET-fix)
	./$(TARGET-fix) 1200

run: $(TARGET)
	$(RUNNER) -n 12 ./$(TARGET) 1200

run-par: $(TARGET)
	$(RUNNER) -n 12 -hostfile $(HOSTFILE) ./$(TARGET)

show: $(PDFTARGET)
	evince $<

$(PDFTARGET): $(PDFSRC)
	$(PDFCC) $<