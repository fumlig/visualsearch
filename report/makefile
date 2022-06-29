SHELL=/bin/bash
PDFVIEWER = xdg-open
TEXMAINFILE = report.tex
PDFNAME = $(shell basename -s.tex $(TEXMAINFILE))
MKLATEX = latexmk
MKLATEXOPTS = -pdf -pdflatex="xelatex -interaction=nonstopmode" -use-make

all: $(PDFNAME).pdf

$(PDFNAME).pdf: $(TEXMAINFILE) **.tex
	$(MKLATEX) $(MKLATEXOPTS) $<

view: $(PDFNAME).pdf
	$(PDFVIEWER) $(PDFNAME).pdf &

watch:
	while true; do make --silent; sleep 2; done

clean:
	$(MKLATEX) -CA
	rm -f $(PDFNAME).pdf *.{aux,bbl,bcf,blg,cb,fdb_latexmk,fls,lof,log,lot,out,run.xml,rel,synctex.gz,toc}
