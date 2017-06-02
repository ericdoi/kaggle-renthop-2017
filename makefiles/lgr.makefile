# Select feature set
#include feature.n6.makefile
include feature.esb4x4.makefile
#include feature.img1.makefile

# todo: refactor
# include base.makefile
# FEATURE_NAME := n6aimg1
# C := 0.085
# FEATURE_TRN := $(DIR_FEATURE)/$(FEATURE_NAME).trn.sps
# FEATURE_TST := $(DIR_FEATURE)/$(FEATURE_NAME).tst.sps
# FEATURE_MAP := $(DIR_FEATURE)/$(FEATURE_NAME).fmap


SOLVER := sag
MULTICLASS := multinomial
#C := 0.25
C := 1

ALGO_NAME := lgr_$(SOLVER)_$(MULTICLASS)_$(C)
MODEL_NAME := $(ALGO_NAME)_$(FEATURE_NAME)

PREDICT_VAL := $(DIR_VAL)/$(MODEL_NAME).val.yht
PREDICT_TST := $(DIR_TST)/$(MODEL_NAME).tst.yht
FEATURE_IMP := $(DIR_MODEL)/$(MODEL_NAME).imp.csv

SUBMISSION_TST := $(DIR_SUB)/$(MODEL_NAME).sub.csv
SUBMISSION_TST_GZ := $(DIR_SUB)/$(MODEL_NAME).sub.csv.gz

all: submission
#validation: $(METRIC_VAL)
submission: $(SUBMISSION_TST)

$(PREDICT_TST) $(PREDICT_VAL) $(FEATURE_IMP): $(FEATURE_TRN) $(FEATURE_TST) $(FEATURE_MAP) \
											| $(DIR_VAL) $(DIR_TST)
	python $(DIR_SRC)/train_predict_lgr.py \
	--train-feature-file $(FEATURE_TRN) \
	--test-feature-file $(FEATURE_TST) \
	--feature-map-file $(FEATURE_MAP) \
	--predict-valid-file $(PREDICT_VAL) \
	--predict-test-file $(PREDICT_TST) \
	--feature-importance-file $(FEATURE_IMP) \
	--solver $(SOLVER) \
	--multiclass $(MULTICLASS) \
	--C $(C)

$(SUBMISSION_TST): $(PREDICT_TST) | $(DIR_SUB)
	paste -d, $(ID_TST) $< > $@

$(SUBMISSION_TST_GZ): $(SUBMISSION_TST)
	gzip $<