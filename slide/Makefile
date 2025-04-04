ifeq ($(OS), Windows_NT)
	RM = del
else
	RM = rm
endif

slide: slide.tex reference.bib Makefile
	xelatex slide.tex
	bibtex slide.aux
	xelatex slide.tex
	xelatex slide.tex

clean:
	$(RM) slide-blx.bib
	$(RM) slide.aux
	$(RM) slide.log
	$(RM) slide.nav
	$(RM) slide.out
	$(RM) slide.run.xml
	$(RM) slide.snm
	$(RM) slide.toc
	$(RM) slide.bbl
	$(RM) slide.blg