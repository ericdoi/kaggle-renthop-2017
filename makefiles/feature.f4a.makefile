include base.makefile

FEATURE_NAME := f4a

FEATURE_TRN := $(DIR_FEATURE)/$(FEATURE_NAME).trn.sps
FEATURE_TST := $(DIR_FEATURE)/$(FEATURE_NAME).tst.sps
FEATURE_MAP := $(DIR_FEATURE)/$(FEATURE_NAME).fmap

build: $(FEATURE_TRN) $(FEATURE_TST) $(ID_TST) $(Y_TRN) $(Y_TST)

$(FEATURE_TRN) $(FEATURE_TST) $(FEATURE_MAP) $(ID_TST) $(Y_TRN) $(Y_TST): $(DATA_TRN) $(DATA_TST) | $(DIR_FEATURE)
	python $(DIR_SRC)/generate_f4a.py \
	--train-file $(DATA_TRN) \
	--test-file $(DATA_TST) \
	--train-feature-file $(FEATURE_TRN) \
	--test-feature-file $(FEATURE_TST) \
	--feature-map-file $(FEATURE_MAP) \
	--test-id-file $(ID_TST) \
	--train-y-file $(Y_TRN) \
	--test-y-file $(Y_TST)
