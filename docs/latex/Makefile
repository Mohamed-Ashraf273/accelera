LATEXMK ?= latexmk
PDFLATEX ?= pdflatex
MAIN := report

.PHONY: pdf clean

pdf:
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex; \
	else \
		$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex; \
		$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex; \
	fi

clean:
	rm -f $(MAIN).aux $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot
