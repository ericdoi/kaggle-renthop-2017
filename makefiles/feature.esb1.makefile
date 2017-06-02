# Adapted from https://github.com/jeongyoonlee/kaggler-template
include base.makefile

FEATURE_NAME := esb1

BASE_MODELS := xgb_10000_6_0.01_f1 \
	           lgr_sag_multinomial_0.25_n1

PREDICTS_TRN := $(foreach m, $(BASE_MODELS), $(DIR_VAL)/$(m).val.yht)
PREDICTS_TST := $(foreach m, $(BASE_MODELS), $(DIR_TST)/$(m).tst.yht)

FEATURE_TRN := $(DIR_FEATURE)/$(FEATURE_NAME).trn.csv
FEATURE_TST := $(DIR_FEATURE)/$(FEATURE_NAME).tst.csv
FEATURE_MAP := $(DIR_FEATURE)/$(FEATURE_NAME).fmap

$(FEATURE_MAP): | $(DIR_FEATURE)
	python ../src/create_fmap_esb.py --base-models $(BASE_MODELS) \
                                     --feature-map-file $@

$(FEATURE_TRN): $(Y_TRN) $(PREDICTS_TRN) | $(DIR_FEATURE)
	paste -d, $^ | tail -n +2 | tr -d '\r' > $@

$(FEATURE_TST): $(Y_TST) $(PREDICTS_TST) | $(DIR_FEATURE)
	paste -d, $^ | tail -n +2 | tr -d '\r' > $@


clean:: clean_$(FEATURE_NAME)

clean_$(FEATURE_NAME):
	-rm $(FEATURE_TRN) $(FEATURE_TST)