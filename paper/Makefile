SHELL := /bin/bash
STYLE_DIR := $(shell dirname "$$(find .. -name aaai2027.sty -print -quit)")
BIBTEX := $(shell if command -v bibtex >/dev/null 2>&1; then command -v bibtex; \
	elif command -v bibtex.original >/dev/null 2>&1; then command -v bibtex.original; \
	elif command -v bibtex8 >/dev/null 2>&1; then command -v bibtex8; fi)

.PHONY: all main supplement checklist draft clean figures reproduce preflight

all: main supplement checklist

main:
	@if [[ -z "$(STYLE_DIR)" || "$(STYLE_DIR)" == "." ]]; then \
	  echo "Could not locate aaai2027.sty under the repository."; exit 1; \
	fi
	@if [[ -z "$(BIBTEX)" ]]; then echo "Could not locate a BibTeX executable."; exit 1; fi
	TEXINPUTS=.:$(STYLE_DIR): BSTINPUTS=.:$(STYLE_DIR): pdflatex -interaction=nonstopmode -halt-on-error main.tex
	TEXINPUTS=.:$(STYLE_DIR): BSTINPUTS=.:$(STYLE_DIR): $(BIBTEX) main
	TEXINPUTS=.:$(STYLE_DIR): BSTINPUTS=.:$(STYLE_DIR): pdflatex -interaction=nonstopmode -halt-on-error main.tex
	TEXINPUTS=.:$(STYLE_DIR): BSTINPUTS=.:$(STYLE_DIR): pdflatex -interaction=nonstopmode -halt-on-error main.tex

supplement: main
	@if [[ -z "$(STYLE_DIR)" || "$(STYLE_DIR)" == "." ]]; then \
	  echo "Could not locate aaai2027.sty under the repository."; exit 1; \
	fi
	@if [[ -z "$(BIBTEX)" ]]; then echo "Could not locate a BibTeX executable."; exit 1; fi
	TEXINPUTS=.:$(STYLE_DIR): BSTINPUTS=.:$(STYLE_DIR): pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
	TEXINPUTS=.:$(STYLE_DIR): BSTINPUTS=.:$(STYLE_DIR): $(BIBTEX) supplement
	TEXINPUTS=.:$(STYLE_DIR): BSTINPUTS=.:$(STYLE_DIR): pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
	TEXINPUTS=.:$(STYLE_DIR): BSTINPUTS=.:$(STYLE_DIR): pdflatex -interaction=nonstopmode -halt-on-error supplement.tex

checklist:
	@if [[ -z "$(STYLE_DIR)" || "$(STYLE_DIR)" == "." ]]; then \
	  echo "Could not locate aaai2027.sty under the repository."; exit 1; \
	fi
	TEXINPUTS=.:$(STYLE_DIR): pdflatex -interaction=nonstopmode -halt-on-error checklist.tex

figures:
	cd .. && PYTHONPATH=src python experiments/run_all.py
	cd .. && PYTHONPATH=src SYNTHETIC_REPETITIONS=30 BOOTSTRAP_REPETITIONS=20 OLS_REPETITIONS=5000 \
	  python experiments/run_submission.py --stage all

reproduce: figures all

draft:
	@if [[ -z "$(BIBTEX)" ]]; then echo "Could not locate a BibTeX executable."; exit 1; fi
	pdflatex -interaction=nonstopmode -halt-on-error draft.tex
	$(BIBTEX) draft
	pdflatex -interaction=nonstopmode -halt-on-error draft.tex
	pdflatex -interaction=nonstopmode -halt-on-error draft.tex
	pdflatex -interaction=nonstopmode -halt-on-error supplement_draft.tex
	$(BIBTEX) supplement_draft
	pdflatex -interaction=nonstopmode -halt-on-error supplement_draft.tex
	pdflatex -interaction=nonstopmode -halt-on-error supplement_draft.tex

preflight: all
	@pages=$$(pdfinfo main.pdf | awk '/Pages/{print $$2}'); \
	  test "$$pages" -le 7 || { echo "main.pdf has $$pages pages"; exit 1; }
	@for file in main.pdf supplement.pdf checklist.pdf; do \
	  if pdffonts "$$file" | grep -q 'Type 3'; then \
	    echo "Type-3 font detected in $$file"; exit 1; \
	  fi; \
	done
	@! grep -Eq 'Overfull|undefined references|Citation.*undefined|Reference.*undefined' \
	  main.log supplement.log checklist.log

clean:
	rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.synctex.gz \
	  draft.pdf main.pdf supplement.pdf supplement_draft.pdf checklist.pdf ReproducibilityChecklist.pdf
