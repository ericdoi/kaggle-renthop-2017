# Select feature set
#include feature.f6a.makefile
#include feature.img1.makefile
include feature.esb4x4.makefile

#FEATURE_NAME := f6

# todo: refactor to use makefile for fpq instead of python
# include base.makefile
# FEATURE_NAME := f6img1
# FEATURE_TRN := $(DIR_FEATURE)/$(FEATURE_NAME).trn.sps
# FEATURE_TST := $(DIR_FEATURE)/$(FEATURE_NAME).tst.sps
# FEATURE_MAP := $(DIR_FEATURE)/$(FEATURE_NAME).fmap

# Ensemble
N_ROUNDS := 10000
DEPTH := 1
LRATE := 0.01

# Best single
# N_ROUNDS := 10000
# DEPTH := 6
# LRATE := 0.02

# Shallow
# N_ROUNDS := 10000
# DEPTH := 3
# LRATE := 0.3

# fpq
#N_ROUNDS := 10001
#DEPTH := 6
#LRATE := 0.01

ALGO_NAME := xgb_$(N_ROUNDS)_$(DEPTH)_$(LRATE)
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
	python $(DIR_SRC)/train_predict_xgb.py \
	--train-feature-file $(FEATURE_TRN) \
	--test-feature-file $(FEATURE_TST) \
	--feature-map-file $(FEATURE_MAP) \
	--predict-valid-file $(PREDICT_VAL) \
	--predict-test-file $(PREDICT_TST) \
	--feature-importance-file $(FEATURE_IMP) \
	--depth $(DEPTH) \
	--lrate $(LRATE) \
	--n-rounds $(N_ROUNDS)

$(SUBMISSION_TST): $(PREDICT_TST) | $(DIR_SUB)
	paste -d, $(ID_TST) $< > $@

$(SUBMISSION_TST_GZ): $(SUBMISSION_TST)
	gzip $<