import os
import sys
sys.path.append(os.path.join('networks'))
from networks.CosFace import CosFace, CosFace1
from networks.SphereFace import SphereFace, SphereFace1
from networks.ArcFace import ArcFace, ArcFace1


def getmodel(face_model, **kwargs):
	"""
		select the face model according to its name
		:param face_model: string
		:param FLAGS: a tf FLAGS (should be replace later)
		:param is_use_crop: boolean, whether the network accepted cropped images or uncropped images
		:loss_type: string, the loss to generate adversarial examples
		return:
		a model class
	"""
	img_shape = (112, 112)
	if face_model == 'CosFace':
		model = CosFace1(**kwargs)
		img_shape = (112, 96)
	elif face_model == 'SphereFace':
		model = SphereFace1(**kwargs)
		img_shape = (112, 96)
	elif face_model == 'ArcFace':
		model = ArcFace1()
	else:
		raise Exception
	return model, img_shape
