LATEXMK ?= latexmk
PDFLATEX ?= lualatex
MAIN := gp_book
PDF := gp_book.pdf

.PHONY: pdf clean

pdf:
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) -lualatex -interaction=nonstopmode -halt-on-error $(MAIN).tex; \
	else \
		$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex; \
		$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex; \
	fi
	@test -f $(PDF)
	@rm -f $(MAIN).aux $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot

clean:
	rm -f $(MAIN).aux $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot $(PDF)
