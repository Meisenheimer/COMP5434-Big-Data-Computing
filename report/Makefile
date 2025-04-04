ifeq ($(OS), Windows_NT)
	RM = del
else
	RM = rm
endif

report: report.tex reference.bib Makefile
	xelatex report.tex
	bibtex report.aux
	xelatex report.tex
	xelatex report.tex

clean:
	$(RM) report.aux
	$(RM) report.bbl
	$(RM) report.blg
	$(RM) report.log
	$(RM) report.out