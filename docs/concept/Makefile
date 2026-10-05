.PHONY: all

all: agfl_concept.pdf current_pipeline.pdf spatial_models.pdf tensor_temporal.pdf supervised_tucker.pdf learned_covariance.pdf

agfl_concept.pdf: agfl_concept.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error agfl_concept.tex

current_pipeline.pdf: current_pipeline.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error current_pipeline.tex

spatial_models.pdf: spatial_models.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error spatial_models.tex

tensor_temporal.pdf: tensor_temporal.tex tensor_temporal_results.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error tensor_temporal.tex

supervised_tucker.pdf: supervised_tucker.tex supervised_tucker_results.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error supervised_tucker.tex

learned_covariance.pdf: learned_covariance.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error learned_covariance.tex
