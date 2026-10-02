.PHONY: all

all: agfl_concept.pdf

agfl_concept.pdf: agfl_concept.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error agfl_concept.tex
