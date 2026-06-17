LATEXMK ?= latexmk
PDFLATEX ?= pdflatex
MAIN := main
PDF := thesis.pdf

.PHONY: pdf clean

pdf:
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex; \
	else \
		$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex; \
		$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex; \
	fi
	cp $(MAIN).pdf $(PDF)
	rm -f $(MAIN).pdf

clean:
	rm -f $(MAIN).aux $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot \
	      $(MAIN).pdf $(PDF)
