PYTHON ?= python
INTERACTIONS := none etym_deriv etym_phon etym_sem phon_sem deriv_sem deriv_phon

.PHONY: all analysis whitebox cv grid-search nn figures clean

## Everything except the neural networks and the grid search
all: analysis whitebox cv figures

## Dataset statistics, feature correlation, semantic gender bias, feature importance
analysis:
	$(PYTHON) scripts/descriptive_stats.py
	$(PYTHON) scripts/feature_correlation.py
	$(PYTHON) scripts/semantic_gender_bias.py
	$(PYTHON) scripts/feature_importance.py

## Interpretable models, single 80/20 split (Table 5, Figure 1)
whitebox:
	$(PYTHON) scripts/predict_gender.py
	$(PYTHON) scripts/predict_inflection.py

## Interpretable models, 5-fold cross-validation (Table 6)
cv:
	$(PYTHON) scripts/predict_gender_cv.py
	$(PYTHON) scripts/predict_inflection_cv.py

## Semantic / Random Forest grid search, 540 configurations (Appendix B)
grid-search:
	$(PYTHON) scripts/grid_search_semantic_rf.py

## Neural networks with feature interactions (Table 3); requires requirements-nn.txt
nn:
	for task in gender inflection; do \
	  for interaction in $(INTERACTIONS); do \
	    $(PYTHON) neural_network/train_interaction.py --task $$task --interaction $$interaction || exit 1; \
	  done; \
	done

## Figure 1 and Markdown versions of Tables 3, 5 and 6
figures:
	$(PYTHON) scripts/make_figure.py
	$(PYTHON) scripts/make_tables.py

clean:
	rm -rf neural_network/checkpoints
