import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from data_modules import FlowDataModule


class FlowClassifier(pl.LightningModule):
	""" Pytorch Lightning wrapper for classifying network flows

	Args:
		network: pytorch model
		learning_rate: _description_. Defaults to 1e-3.
		reduce_lr_on_plateau (bool): Defaults to False.
		reduce_lr_factor (float): Defaults to 0.1.
		patience (int): Reduce LR patience. Defaults to 10.
	"""
	def __init__(self, network,
					learning_rate=1e-3, reduce_lr_on_plateau=False, 
					reduce_lr_factor=0.1, patience=10,
					pos_weight=None):
		super().__init__()
		self.network = network
		self.learning_rate = learning_rate
		self.reduce_lr_on_plateau = reduce_lr_on_plateau
		self.reduce_lr_factor = reduce_lr_factor
		self.patience = patience
		self.pos_weight = pos_weight

	def forward(self, x):
		if isinstance(x, dict):
			x = x['features']
		return self.network(x)

	def training_step(self, batch, batch_idx=None):
		y = batch['label']
		x = batch['features']
		y_hat = self(x)
		loss = F.binary_cross_entropy_with_logits(
			y_hat, 
			y.unsqueeze(1).float(),
			pos_weight=self.pos_weight
		)
		self.log("train_loss", loss)
		return {'loss': loss}

	def validation_step(self, batch, batch_idx=None):
		y = batch['label']
		x = batch['features']
		y_hat = self(x)
		loss = F.binary_cross_entropy_with_logits(
			y_hat, 
			y.unsqueeze(1).float(),
			pos_weight=self.pos_weight
		)
		self.log("val_loss", loss, on_epoch=True, prog_bar=True)

	def configure_optimizers(self):
		if self.reduce_lr_on_plateau:
			optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
				optimizer, 
				factor=self.reduce_lr_factor, 
				patience=self.patience,
				verbose=True
			)
			return {
				'optimizer': optimizer,
				'lr_scheduler': {
					'scheduler': scheduler,
					'monitor': 'val_loss',
				}
			}
		else:
			return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class DenseNet(nn.Module):
	def __init__(self, input_size, layer_sizes, output_size, 
					output_activation='linear', dropout=0.5):
		super(DenseNet, self).__init__()
		assert len(layer_sizes) > 0

		layers = []

		for i, n_hidden in enumerate(layer_sizes):
			if i == 0:
				layers.append(nn.Linear(input_size, n_hidden))
			else:
				layers.append(nn.Linear(layer_sizes[i-1], n_hidden))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(p=dropout))
		
		layers.append(nn.Linear(layer_sizes[-1], output_size))

		if output_activation == 'sigmoid':
			layers.append(nn.Sigmoid())
		elif output_activation != 'linear':
			raise ValueError('output_activation must be either "sigmoid" or "linear"')

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)


if __name__ == '__main__':
	# Basic test
	data_module = FlowDataModule(
		label_type='binary',
		use_protocol=True
	)
	data_module.setup()
	batch = data_module.train_dataloader().__iter__().__next__()

	net = DenseNet(
		input_size=data_module.train_dataset.features.shape[1],
		output_size=1,
		layer_sizes=[32, 32],
	)
	model = FlowClassifier(net)

	batch_logits = model(batch['features'])
