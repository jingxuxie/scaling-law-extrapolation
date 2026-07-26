SHELL := /bin/bash
STYLE_DIR := $(shell dirname "$$(find .. -name aaai2027.sty -print -quit)")
BIBTEX := $(shell if command -v bibtex >/dev/null 2>&1; then command -v bibtex; \
	elif command -v bibtex.original >/dev/null 2>&1; then command -v bibtex.original; \
	elif command -v bibtex8 >/dev/null 2>&1; then command -v bibtex8; fi)

.PHONY: all main supplement draft clean figures

all: main supplement

main:
	@if [[ -z "$(STYLE_DIR)" || "$(STYLE_DIR)" == "." ]]; then \
	  echo "Could not locate aaai2027.sty under the repository."; exit 1; \
	fi
	@if [[ -z "$(BIBTEX)" ]]; then echo "Could not locate a BibTeX executable."; exit 1; fi
	TEXINPUTS=.:$(STYLE_DIR): BSTINPUTS=.:$(STYLE_DIR): pdflatex -interaction=nonstopmode -halt-on-error main.tex
	TEXINPUTS=.:$(STYLE_DIR): BSTINPUTS=.:$(STYLE_DIR): $(BIBTEX) main
	TEXINPUTS=.:$(STYLE_DIR): BSTINPUTS=.:$(STYLE_DIR): pdflatex -interaction=nonstopmode -halt-on-error main.tex
	TEXINPUTS=.:$(STYLE_DIR): BSTINPUTS=.:$(STYLE_DIR): pdflatex -interaction=nonstopmode -halt-on-error main.tex

supplement:
	@if [[ -z "$(STYLE_DIR)" || "$(STYLE_DIR)" == "." ]]; then \
	  echo "Could not locate aaai2027.sty under the repository."; exit 1; \
	fi
	TEXINPUTS=.:$(STYLE_DIR): BSTINPUTS=.:$(STYLE_DIR): pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
	TEXINPUTS=.:$(STYLE_DIR): BSTINPUTS=.:$(STYLE_DIR): pdflatex -interaction=nonstopmode -halt-on-error supplement.tex

draft:
	@if [[ -z "$(BIBTEX)" ]]; then echo "Could not locate a BibTeX executable."; exit 1; fi
	pdflatex -interaction=nonstopmode -halt-on-error draft.tex
	$(BIBTEX) draft
	pdflatex -interaction=nonstopmode -halt-on-error draft.tex
	pdflatex -interaction=nonstopmode -halt-on-error draft.tex
	pdflatex -interaction=nonstopmode -halt-on-error supplement_draft.tex
	pdflatex -interaction=nonstopmode -halt-on-error supplement_draft.tex

figures:
	cd .. && PYTHONPATH=src python experiments/run_all.py

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.pdf
