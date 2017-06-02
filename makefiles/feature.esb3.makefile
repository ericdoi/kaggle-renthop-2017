# Adapted from https://github.com/jeongyoonlee/kaggler-template
include base.makefile

FEATURE_NAME := esb3

BASE_MODELS := lgr_sag_multinomial_1_img1 \
	           xgb_10000_3_0.3_img1 \
	           xgb_10000_3_0.3_f6img1 \
	           lgr_sag_multinomial_0.085_n6aimg1


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