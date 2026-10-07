LATEXMK ?= latexmk

.PHONY: all clean
all: report.pdf

report.pdf: report.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error report.tex

clean:
	$(LATEXMK) -C report.tex
