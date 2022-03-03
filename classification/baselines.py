"""
Train baseline conventional ML models.

Saves jsons with model performance info in ./baseline_res/{model_type}
"""

from datetime import datetime
import os
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from data_modules import FlowDataModule


if __name__ == '__main__':
	# Training options
	model_desc = None   # for naming saved results, if None uses timestamp
	train_opts = {
		'use_protocol': True,
		'model_type': 'rf', # 'rf', 'ada', 'xgb', 'svm'
		'label_type': 'binary', # 'binary', 'both'
		'scale': False,	# may want to set to True when using SVM
	}
	rf_opts = {
		'n_estimators': 1000,
		'criterion': 'gini',
		'max_depth': None,
		'max_features': 'auto',
	}
	ada_opts = {
		'n_estimators': 100,
		'learning_rate': .8,
	}
	xgb_opts = {
		'n_estimators': 1000,
		'learning_rate': .2,
		'tree_method': 'gpu_hist',
		'max_depth': None,
	}
	svm_opts = {
		'kernel': 'rbf',
		'tol': 1e-6,
		'C': .7,
		'class_weight': None # 'balanced'
	}

	# Load data
	data_module = FlowDataModule(
		label_type=train_opts['label_type'],
		use_protocol=train_opts['use_protocol']
	)
	data_module.setup()
	train_data = data_module.train_dataset
	val_data = data_module.val_dataset
	test_data = data_module.test_dataset

	# Create model
	if train_opts['model_type'] == 'rf':
		model_opts = rf_opts
		model = RandomForestClassifier(
			verbose=2, n_jobs=-1, **rf_opts)
	elif train_opts['model_type'] == 'ada':
		model_opts = ada_opts
		model = AdaBoostClassifier(**ada_opts)
	elif train_opts['model_type'] == 'xgb':
		model_opts = xgb_opts
		if train_opts['label_type'] == 'binary':
			model = xgb.XGBClassifier(
				verbosity=2, n_jobs=-1, use_label_encoder=False, 
				objective='binary:logistic',
				**xgb_opts
			)
	elif train_opts['model_type'] == 'svm':
		model_opts = svm_opts
		if train_opts['scale']:
			model = make_pipeline(
				StandardScaler(), 
				SVC(verbose=True, probability=True, **svm_opts)
			)
		else:
			model = SVC(verbose=True, probability=True, **svm_opts)
	else:
		raise NotImplementedError

	# Train model
	model.fit(train_data.features.values, train_data.labels.values)

	# Evaluate model
	val_preds = model.predict_proba(val_data.features.values)
	test_preds = model.predict_proba(test_data.features.values)

	val_true = val_data.labels.values
	test_true = test_data.labels.values

	if train_opts['label_type'] == 'binary':
		val_preds_bin = np.argmax(val_preds, axis=1)
		test_preds_bin = np.argmax(test_preds, axis=1)

		val_preds = val_preds[:, 1]
		test_preds = test_preds[:, 1]

		val_class_report = classification_report(val_true, val_preds_bin, output_dict=True)
		test_class_report = classification_report(test_true, test_preds_bin, output_dict=True)

		val_cm = confusion_matrix(val_true, val_preds_bin)
		test_cm = confusion_matrix(test_true, test_preds_bin)

		val_prc = precision_recall_curve(val_true, val_preds)
		test_prc = precision_recall_curve(test_true, test_preds)

		save_dict = train_opts
		save_dict.update(model_opts)

		save_dict['val_class_report'] = val_class_report
		save_dict['test_class_report'] = test_class_report
		save_dict['val_cm'] = val_cm.tolist()
		save_dict['test_cm'] = test_cm.tolist()
		save_dict['val_prc'] = [a.tolist() for a in val_prc]
		save_dict['test_prc'] = [a.tolist() for a in test_prc]
		save_dict['val_preds'] = val_preds.tolist()
		save_dict['test_preds'] = test_preds.tolist()
		save_dict['val_true'] = val_true.tolist()
		save_dict['test_true'] = test_true.tolist()
		save_dict['val_preds_bin'] = val_preds_bin.tolist()	# binary version of predictions
		save_dict['test_preds_bin'] = test_preds_bin.tolist()

	else:
		raise NotImplementedError

	# Save results
	if model_desc is None:
		model_desc = int(datetime.now().timestamp())
	res_save_dir = os.path.join('baseline_res', train_opts['model_type'])

	if not os.path.exists(res_save_dir):
		os.makedirs(res_save_dir)

	with open(os.path.join(res_save_dir, '{}.json'.format(model_desc)), 'w') as f:
		json.dump(save_dict, f, indent=4)




