MAIN = main
ZIP  = CS5478_Team34_Report_Overleaf.zip

.PHONY: all pdf clean overleaf

all: pdf

pdf:
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

overleaf:
	rm -f $(ZIP)
	zip -r $(ZIP) main.tex neurips_2024.sty references.bib sections/ figures/ -x "*.DS_Store"

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz
