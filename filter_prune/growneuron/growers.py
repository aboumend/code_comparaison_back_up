# coding=utf-8
# Copyright 2022 GradMax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


## ce fichier implemente les differentes methodes de growing (gradmax, gradmaxopt, firefly, random)
## il fait ensuite appelle aux methodes de layers.py pour creer les layers resultants du growth selon une certaine methode (random, gradmax...)
## dans le fichier layers.py il y a des callbacks, c'est depuis le fichier growers.py qu'on ajoute/manipule les callbacks
## le calcul de svd se fait dans la methode get_growth_directions

"""This module implements various growing algorithms.
"""
import functools
import logging
#import growneuron.layers as glayers
import layers as glayers
import numpy as np
import random
import tensorflow as tf


class LayerGrower():
  """Base class for growing layer algorithms.

  Subclasses should implement grow_neurons.
    grad_fn: Should return list of variables and return aggregated gradients.
    grow_layers: 2 layers. First is the
      one we are deleting filters from and the second is the layer that consumes
      neurons from the first layer. However in some architectures there could
      be some layers in between that transforms channel-wise information
      independetly: like Batchnorm and depth-wise convolutions. In such cases,
      we grow identity neurons for the layers inbetween the first and last.
  """
  ## certain des attributs sont fixe depuis le fichier depuis lequel on lance le training
  #le compile_fn, loss_fn et strategy sont aussi manipules dans le fichier updaters.py
  #pour le strategy c'est je pense lie a tf.distribute.Strategy
  #tf.distibute.Strategy est une api qui permet de repartir le training sur plusieurs gpu/tpu/equipement
  #le lodd_fn normalement on le definit dans le fichier main ou on lance le training
  
  compile_fn = lambda: None #parce que dans tensorflow on fait un model.compile(ou on donne en argument le loss, optimizer, metrics evaluation) avant le model.fit
  loss_fn = lambda x: x

  def grow_neurons(self, grow_layers, batch_data, **kwargs): #sera implemente dans les sous-classes, la c'est juste la classe de base
    raise NotImplementedError()

## Random growing
class AddRandom(LayerGrower): #herite de layergrower, doit implementer grow_neurons en faisant un random growing
  def grow_neurons(self, grow_layers, batch_data, n_to_remove=0, list_filters_to_remove=None):
    #del batch_data #supprime l'objet batch_data car pour le random grow on n'a pas besoin du dataset pour des calculs qu'impose l'approche ici c'est random
    
    
    
    for i, layer in enumerate(grow_layers): 
      #pour ce que veut dire grow_layers voir le comment de la classe LayerGrower(en gros c'est la liste des layer concernes par le growth)
      if i == 0:
        # First layer: donc normalement celui auquel on rajoute les nouveaux neurones
        layer.add_neurons(n_to_remove, list_filters_to_remove=list_filters_to_remove, is_outgoing=False) #add_neurons est definie dans le fichier layers.py, c'est la methode qui englobe le processus d'ajout de neurone a un layer
                                            #en gros ca permet de creer le layer resultat du growth (layer avant growth + nouveau neurones)
      elif i == (len(grow_layers) - 1):
        # Last layer: layer qui consomme/recoit des neurones de first layer
        layer.add_neurons(n_to_remove, list_filters_to_remove=list_filters_to_remove, is_outgoing=True)
      else: #layer de normalization/depthwiseconv qui sont entre ce qu'on a appele first et last layer, on leur ajoute des identity neurones selon la methode
            #add_neurons_identity du fichier layers.py
        if isinstance(layer, glayers.GrowLayer):
          layer.add_neurons_identity(n_to_remove)

  def prune_filters_l1_norm(self, grow_layers, n_to_remove=0):
    #grow_layers contient le layer à élaguer, le layer qui le suit et éventuellement les layers de batchnorm entre les deux
    #on récupère le layer à élaguer (grow_layers[0]) pour calculer la norme de ses filtres
    layer_to_prune=grow_layers[0]
    weights = layer_to_prune.get_weights()[0]
    #print(weights.shape)
    """#on sépare les filtres, on a autant de filtres que d'output channels (denière dimension d'où le -1)
    filters=np.split(np.absolute(weights), weights.shape[-1], axis=len(weights.shape)-1)
    filters=np.array(filters)
    l1_norm= np.sum(filters, axis=(1,2,3))
    print(l1_norm)
    print(filters.shape)
    print(len(filters), filters[0].shape)"""

    #calcul de la norme L1 des filtres séparés selon la dimension de l'output channels donc selon le nombre de filtres (on a autant de normes que de filtres)
    l1_norm=np.sum(np.absolute(weights), axis=(0,1,2))
    
    #print(np.sum(np.absolute(weights[:,:,:,0])))
    #print(len(l1_norm))
    #print(l1_norm)
    #np.argpartition nous permet de récupérer les indices ordonnés selon les valeurs de l1_norm (un argmin mais avec k élements)
    #si on fait un slice list_filters_to_remove[:n_to_remove] on les indices des k (=n_to_remove) plus petite normes donc k filtres à élaguer et c'est ce qu'on retourne
    
    list_filters_to_remove=np.argpartition(l1_norm, n_to_remove)

    #print(list_filters_to_remove)
    #print(list_filters_to_remove[:n_to_remove])
    #sorted_norm=sorted(l1_norm)
    #print(sorted_norm)
    return list_filters_to_remove[:n_to_remove]

  def prune_filters_frob_norm(self, grow_layers, n_to_remove=0):
    #grow_layers contient le layer à élaguer, le layer qui le suit et éventuellement les layers de batchnorm entre les deux
    #on récupère le layer à élaguer (grow_layers[0]) pour calculer la norme de ses filtres
    layer_to_prune=grow_layers[0]
    weights = layer_to_prune.get_weights()[0]
    #print(weights.shape)
    """#on sépare les filtres, on a autant de filtres que d'output channels (denière dimension d'où le -1)
    """

    #calcul de la norme L1 des filtres séparés selon la dimension de l'output channels donc selon le nombre de filtres (on a autant de normes que de filtres)
    frob_norm=np.sqrt(np.sum(np.power(np.absolute(weights),2), axis=(0,1,2)))
    
    
    #np.argpartition nous permet de récupérer les indices ordonnés selon les valeurs de l1_norm (un argmin mais avec k élements)
    #si on fait un slice list_filters_to_remove[:n_to_remove] on les indices des k (=n_to_remove) plus petite normes donc k filtres à élaguer et c'est ce qu'on retourne
    list_filters_to_remove=np.argpartition(frob_norm, n_to_remove)

    #print(list_filters_to_remove)
    #print(list_filters_to_remove[:n_to_remove])
    
    return list_filters_to_remove[:n_to_remove]


  def prune_filters_random(self, grow_layers, n_to_remove=0):
    #grow_layers contient le layer à élaguer, le layer qui le suit et éventuellement les layers de batchnorm entre les deux
    #on récupère le layer à élaguer (grow_layers[0]) pour calculer la norme de ses filtres
    print("random pruning")
    layer_to_prune=grow_layers[0]
    nb_filters = layer_to_prune.get_weights()[0].shape[-1]
    #print(layer_to_prune.get_weights()[0].shape)
    #print(nb_filters)
    
    list_filters_to_remove=random.sample(range(nb_filters), n_to_remove) #pour générer n_to_remove nb sans répétition, on génère entre 0 et nb_filters
    
    print("filters to remove", list_filters_to_remove)
    
    
    return list_filters_to_remove
    
         





def extract_image_patches(x, kernel_size, stride=(1, 1)):
  """Extract convolutional patches from the layer.

  Manual replacement of tf.extract_image_patches, since its gradient cannot
  be evaluated on TPU.

  Args:
    x: batched input data. Size: [batch, in_height, in_width, in_channels]
    kernel_size: Tuple of two integers. Size of kernel.
    stride: Tuple of two integers. Stride size.

  Returns:
    4D Tensor (batch, in_rows, in_cols, patch_size) of extracted patches.
  """
  in_channels = x.get_shape()[3]
  kh, kw = kernel_size
  tile_filter = np.zeros(shape=[kh, kw, in_channels, kh * kw], dtype=np.float32)
  for i in range(kh):
    for j in range(kw):
      tile_filter[i, j, :, i * kw + j] = 1.0

  tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
  output = tf.nn.depthwise_conv2d(
      x, tile_filter_op, strides=[1, *stride, 1], padding='VALID')
  # reshaping below is needed so that 4th dimension of the output can be
  # reshaped into kernel[0] * kernel[1] * in_channels.
  batch, in_rows, in_cols, _ = output.get_shape()
  output = tf.reshape(
      output, shape=[batch, in_rows, in_cols, in_channels, kh * kw])
  output = tf.transpose(output, perm=[0, 1, 2, 4, 3])
  output = tf.reshape(output, [batch, in_rows, in_cols, -1])

  return output
