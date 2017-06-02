include base.makefile

N_ROUNDS := 1000
DEPTH := 4
LRATE := 0.3

FEATURE_NAME := pm2_$(N_ROUNDS)_$(DEPTH)_$(LRATE)

FEATURE_TRN := $(DIR_FEATURE)/$(FEATURE_NAME).trn.sps
FEATURE_TST := $(DIR_FEATURE)/$(FEATURE_NAME).tst.sps
FEATURE_MAP := $(DIR_FEATURE)/$(FEATURE_NAME).fmap

build: $(FEATURE_TRN) $(FEATURE_TST)

$(FEATURE_TRN) $(FEATURE_TST) $(FEATURE_MAP): $(DATA_TRN) $(DATA_TST) | $(DIR_FEATURE)
	python $(DIR_SRC)/generate_price_median_feats.py \
	--train-file $(DATA_TRN) \
	--test-file $(DATA_TST) \
	--train-feature-file $(FEATURE_TRN) \
	--test-feature-file $(FEATURE_TST) \
	--feature-map-file $(FEATURE_MAP) \
	--depth $(DEPTH) \
	--lrate $(LRATE) \
	--n-rounds $(N_ROUNDS)
