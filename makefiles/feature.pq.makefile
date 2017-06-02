include base.makefile

N_ROUNDS := 130
DEPTH := 12
LRATE := 0.2
INT_OR_OHE := ohe

FEATURE_NAME := pq_$(N_ROUNDS)_$(DEPTH)_$(LRATE)_$(INT_OR_OHE)

FEATURE_TRN := $(DIR_FEATURE)/$(FEATURE_NAME).trn.sps
FEATURE_TST := $(DIR_FEATURE)/$(FEATURE_NAME).tst.sps
FEATURE_MAP := $(DIR_FEATURE)/$(FEATURE_NAME).fmap

build: $(FEATURE_TRN) $(FEATURE_TST)

validate: $(DATA_TRN) | $(DIR_FEATURE)
	python $(DIR_SRC)/validate_price_quantiles.py \
	--train-file $(DATA_TRN) \
	--depth $(DEPTH) \
	--lrate $(LRATE) \
	--n-rounds $(N_ROUNDS)

$(FEATURE_TRN) $(FEATURE_TST) $(FEATURE_MAP): $(DATA_TRN) $(DATA_TST) | $(DIR_FEATURE)
	python $(DIR_SRC)/generate_price_quantiles.py \
	--train-file $(DATA_TRN) \
	--test-file $(DATA_TST) \
	--train-feature-file $(FEATURE_TRN) \
	--test-feature-file $(FEATURE_TST) \
	--feature-map-file $(FEATURE_MAP) \
	--depth $(DEPTH) \
	--lrate $(LRATE) \
	--n-rounds $(N_ROUNDS) \
	--int-or-ohe $(INT_OR_OHE)
