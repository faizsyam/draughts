import torch
import torch.nn as nn
import config2
import numpy as np

in_channels = 4 # game (map), mask, current player indicator
map_size = (10, 10)

# class to be used
class CNN_Net(nn.Module):
	def __init__(self):
		super(CNN_Net, self).__init__()
		self.conv_block = Conv_block()
		self.residual_blocks = self.__make_residual_blocks()
		self.value_head = Value_head()
		self.value_capture_head = Value_head()
		self.policy_head = Policy_head()


	def forward(self, x, is_capture):
		# x = x.type(torch.DoubleTensor)
		out = self.conv_block(x)
		out = self.residual_blocks(out)
		
		if is_capture:
			value = self.value_capture_head(out)
			return value
		else:
			value = self.value_head(out)
			policy = self.policy_head(out)
			return value, policy

	# def train():
	# 	return self.m

	def __make_residual_blocks(self):
		blocks = []
		for _ in range(config2.nb_residual_blocks):
			blocks.append(Residual_block())
		return nn.Sequential(*blocks)


	def number_of_trainable_parameters(self):
		return sum([x.numel() for x in self.parameters() if x.requires_grad])

	# def visualize_model(self):
	# 	example_input = torch.randn((5, 3, 8, 8))
	# 	values, policies = model(example_input)

	def message(self, mess):
		self.logger.info(mess)
		print(mess)

	def convertToModelInput(self,binary):
		inputToModel = np.reshape(binary, (4,10,10)) 
		return np.array([inputToModel])

class Policy_head(nn.Module):
	def __init__(self):
		super(Policy_head, self).__init__()
		nb_filters_policy_head = 2
		self.conv = nn.Conv2d(config2.nb_filters_3x3, 
					 		  out_channels=nb_filters_policy_head,
					 		  kernel_size=(1, 1),
					 		  stride=1,
					 		  padding=0,
					 		  bias=True)
		self.bn = nn.BatchNorm2d(nb_filters_policy_head)
		self.relu = nn.ReLU(inplace=True)
		self.fc = nn.Linear(in_features=nb_filters_policy_head * map_size[0] * map_size[1], 
							out_features=570, # << out size here
							bias=True)
		self.softmax = nn.Softmax()


	def forward(self, x):
		out = self.conv(x)
		out = self.bn(out)
		out = self.relu(out)
		out = self.fc(out.view(out.size(0), -1))
		out = self.softmax(out)
		return out


class Value_head(nn.Module):
	def __init__(self):
		super(Value_head, self).__init__()
		nb_filters_value_head = 1
		self.conv = nn.Conv2d(config2.nb_filters_3x3, 
					 		  out_channels=nb_filters_value_head,
					 		  kernel_size=(1, 1),
					 		  stride=1,
					 		  padding=0,
					 		  bias=True)
		self.bn = nn.BatchNorm2d(nb_filters_value_head)
		self.relu1 = nn.ReLU(inplace=True)
		self.fc1 = nn.Linear(in_features=nb_filters_value_head * map_size[0] * map_size[1], 
							 out_features=config2.value_head_hidden_layer_size, 
							 bias=True)
		self.relu2 = nn.ReLU(inplace=True)
		self.fc2 = nn.Linear(in_features=config2.value_head_hidden_layer_size, 
							 out_features=1,
							 bias=True)
		self.tanh = nn.Tanh()


	def forward(self, x):
		out = self.conv(x)
		out = self.bn(out)
		out = self.relu1(out)
		out = self.fc1(out.view(out.size(0), -1))
		out = self.relu2(out)
		out = self.fc2(out)
		out = self.tanh(out)
		return out


class Residual_block(nn.Module):
	def __init__(self):
		super(Residual_block, self).__init__()
		self.conv1 = conv3x3(config2.nb_filters_3x3)
		self.bn1 = nn.BatchNorm2d(config2.nb_filters_3x3)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(config2.nb_filters_3x3)
		self.bn2 = nn.BatchNorm2d(config2.nb_filters_3x3)
		self.relu2 = nn.ReLU(inplace=True)


	def forward(self, x):
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out += identity
		out = self.relu2(out)
		return out



class Conv_block(nn.Module):
	def __init__(self):
		super(Conv_block, self).__init__()
		self.conv = conv3x3(in_channels)
		self.bn = nn.BatchNorm2d(config2.nb_filters_3x3)
		self.relu = nn.ReLU(inplace=True)


	def forward(self, x):
		# print(x)
		# print(x.size())
		out = self.conv(x)
		out = self.bn(out)
		out = self.relu(out)
		return out


def conv3x3(in_channels):
	return nn.Conv2d(in_channels, 
					 out_channels=config2.nb_filters_3x3,
					 kernel_size=(3, 3),
					 stride=1,
					 padding=1,
					 bias=True)