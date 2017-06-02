include base.makefile

FEATURE_NAME := img1
DIR_IMG_FEAT := $(DIR_BUILD)/photo-features/aggs-full

FEATURE_TRN := $(DIR_FEATURE)/$(FEATURE_NAME).trn.sps
FEATURE_TST := $(DIR_FEATURE)/$(FEATURE_NAME).tst.sps
FEATURE_MAP := $(DIR_FEATURE)/$(FEATURE_NAME).fmap

build: $(FEATURE_TRN) $(FEATURE_TST) $(ID_TST)

$(FEATURE_TRN) $(FEATURE_TST) $(FEATURE_MAP) $(ID_TST): $(DATA_TRN) $(DATA_TST) | $(DIR_FEATURE)
	python $(DIR_SRC)/generate_img1.py \
	--aggs-path $(DIR_IMG_FEAT) \
	--train-file $(DATA_TRN) \
	--test-file $(DATA_TST) \
	--train-feature-file $(FEATURE_TRN) \
	--test-feature-file $(FEATURE_TST) \
	--feature-map-file $(FEATURE_MAP)
