PDF=accpipe_para.pdf
SRC=accpipe_para.tex

.PHONY: all clean

all: $(PDF)

$(PDF): $(SRC)
	pdflatex -interaction=nonstopmode -halt-on-error $(SRC)
	pdflatex -interaction=nonstopmode -halt-on-error $(SRC)

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk accpipe_para.pdf $(PDF)
