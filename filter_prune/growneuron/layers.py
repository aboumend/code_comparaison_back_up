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


## ce fichier definit les methodes qui ajoutent des nouveaux poids a des layers existants
## ca definit seulement l'ajout donc comment le layer change mais c'est pas dans ce fichier que l'algo gradmax(avec svd) est implemente
"""GrowLayer wrapper module."""
import numpy as np
import tensorflow as tf

#type layers supportes
SUPPORTED_LAYERS = (tf.keras.layers.Conv2D)


def get_activation_fn(actv_fn):
  ## je pense que ca leur permet de modifier le relu de facon a respecter les suppositions f(0)=0 et f'(0)=1 (f etant la fonction d'activation)
  ## l'argument de la fonction est une chaine de caractere qui est relu1 ou relu2 ou autre donc nom fonction activation
  ## ca retourne la fonction d'activation relu modifie pour respecter les suppositions
  """Activation choices for the layer.

  Args:
    actv_fn: str.

  Returns:
    activation fn
  """
  if actv_fn == 'relu1':
    # This has grad(f(0))=1 instead of 0 (default implementation).===> donc pour avoir f'(0)=1
    return lambda x: tf.math.maximum(x, 0)
  elif actv_fn == 'relu2':
    # This has grad(f(0))=1.
    return lambda x: tf.math.maximum(x, -1)
  else: #si c'est une activation autre que relu
    return tf.keras.activations.get(actv_fn)


class GrowLayer(tf.keras.layers.Wrapper):
  ## certain des attributs sont fixe depuis le fichier depuis lequel on lance le training
  ## selon leur explication la classe permet de rajouter une fonction de rappel (callback) a la forward pass
  ## la fonction de rappel/callback est une fonction passee en argument d'une autre fonction ainsi cette derniere peut utiliser la fonction de rappel
  ## dans keras les callbacks sont une facon de customizer la behaviour du model par exemple on peut les utiliser pour implementer le learning rate schedule
  ## je pense que le but est de faire le grow
  ## selon les explication, la classe permet aux composants de layer.keras de supporter le growing
  ## pour les callbacks et les methodes definies dessus dans cette classe elles sont utilisees dans le fichier growers.py pas dans le fichier actuel
  ## c'est la fonction get_growth_directions du fichier growers.py qui ajoute les callbacks et les manipulent
  """This layer wraps keras.layers in order to support growing.

  This layer allows adding callbacks to the forward pass of a layer that will be
  called with the inputs and outputs of the underlying layer before the
  activations.

  Example Usage:
    ```
    first_layer = GrowLayer(tf.keras.layers.Dense(32))
    ```

  This layer can be used for growing neurons.
    `first_layer.add_neurons_incoming(1, new_weights='zeros')`
  """

  def __init__(self, *args, activation=None, **kwargs):
    if 'name' not in kwargs:
      # args[0] is the wrapped layer
      kwargs['name'] = f'glayer_{args[0].name}'
    super().__init__(*args, **kwargs)
    self.activation = get_activation_fn(activation) #get_activation_fn est definie juste avant cette classe
    self.reset_callbacks() #definie par la suite

  def add_callback(self, name, fn):
    self._callbacks[name] = fn

  def remove_callback(self, name):
    del self._callbacks[name]

  def reset_callbacks(self):
    self._callbacks = {}

  def __call__(self, inputs, *args, **kwargs): #peut etre une redefinition du call de tensorflow, c'est dans call qu'on definit les calculs que doit faire le layer
    outputs = self.layer.__call__(inputs, *args, **kwargs) #la on fait le call classique
    for _, callback_fn in self._callbacks.items(): #on parcourt les callbacks et on les applique
      inputs, outputs = callback_fn(inputs, outputs)
    if self.activation:
      outputs = self.activation(outputs)
    return outputs

  def add_neurons(self, n_to_remove, list_filters_to_remove=None,
                  is_outgoing=False):
    #voir leur explication
    """Adds new neurons and creates a new layer.

    New weights are scaled (if not zero) to have l2-norm equal to the mean
    l2-norm of the existing weights.
    TODO Unify splitting and adding neurons.
    Args:
      n_to_remove: number of filters to remove.
      list_filters_to_remove: filters to remove from layer
      is_outgoing: bool, if true adds outgoing connections from the new neurons
        coming from previous layers. In other words number of neurons in current
        layer stays constant, but they aggregate information from n_to_remove many
        new neurons.
    """
    old_module = self.layer
    #le assert teste si un boolean est a true: si c'est true rien ne se passe sinon on a une exception
    assert old_module.built
    
    assert isinstance(old_module, SUPPORTED_LAYERS) #verifier si c'est un fc layer ou un conv layer
    self.layer = grow_new_layer(old_module, n_to_remove, list_filters_to_remove,
                                is_outgoing=is_outgoing) #appel de la fonction qui fait le travail du growing
    #on modifie self.layer donc on remplace le layer existant (oldmodule) par un autre layer a qui on a ajoute des neurones (a subit le grow) 

  #pour l'utilite de add_neurons_identity dans le fichier growers.py ils expliquent que s'il y a des layer de nromalization/depthwiseconv entre deux layer concernes par le growth
  # (les layers concernes sont le layers qui subit le growth et le layers suivant car recoit les nouveaux poids), les layers entre les deux concernes
  #(notamment ceux de la normalization/depthwiseconv) on leur fait un identity growing
  #quand on dit identite je pense c'est au sens (0,1)
  def add_neurons_identity(self, n_to_remove):
    """Adds identity neurons for various layer types.

    Args:
      n_to_remove: number of filters to remove.
    """
    old_module = self.layer
    assert old_module.built
    
    #comme on rajoute des identity neurones, ils retrouvent d'abord quel est le type du layer (en utilisant isinstance(oldmodule, type layer))
    #puis ils appellent une fonction qui fait le grow selon le type du layer en passant n_to_remove (nb neurones a ajouter) en argument
    if isinstance(old_module, tf.keras.layers.BatchNormalization):
      self.layer = grow_new_bn_layer(old_module, n_to_remove)
    elif isinstance(old_module, tf.keras.layers.LayerNormalization): #normalization differente de la batch norm voir la doc si necessaire
      self.layer = grow_new_ln_layer(old_module, n_to_remove)
    elif isinstance(old_module, tf.keras.layers.DepthwiseConv2D): #cas du mobilenet, ils definissent l'identite parce que dans l'article ils expliquent que lors du
                                                                  #grow d'un depthwise convlayer, la depthwise conv est initialise comme etant l'identite
      self.layer = grow_new_dw_layer(old_module, n_to_remove)
    else:
      raise ValueError(f'layer: {old_module} of {type(old_module)} is not '
                       'supported.') #si oldmodule ne fait pas partie des layer supporte donc conv et fclayer


def grow_new_layer(old_module, n_to_remove, list_filters_to_remove, is_outgoing=False):
  #ca sert a creer un nouveau layer mais pas au sens on a ajoute un nouveau, c'est plutot creer un nouveau layer auquel on a ajoute les neurones
  #donc ca sert a remplacer un certain layer par ce certain layer auquel on ajoute les nouveaux neurones
  """Creates new layer after removing incoming or outgoing connections.

  Args:
    old_module: Old layer. One of layers.SUPPORTED_LAYERS.
    n_to_remove: number of filters to remove.
    list_filters_to_remove: filters to be removed from layer.
    is_outgoing: bool, True if the outgoing connections of the new neurons are
      being added to the next layer. In this case, no new neurons are generated;
      instead existing neurons receive new incoming connections.
  Returns:
    layer of same type as the old_module.
  """
  old_weights = old_module.get_weights()[0] #on prend les poids de oldmodule (layer avant growth)
  shape_axis = -2 if is_outgoing else -1 #pour dire est-ce que c'est les input ou les output channels qu'on modifie
                                         #on modifie les output channels pour le layer pour lequel on supprime un filtre
                                         #on modifie les input channels pour le layer qui suit celui pour lequel on supprime un filtre

  shape_new = list(old_weights.shape)
  shape_new[shape_axis] = shape_new[shape_axis] - n_to_remove #ca doit etre pour changer les dimensions et ajouter les nouveaux neurones 
  #on commence par spliter les poids selon la dimension input/output channels pour separer les filtres et pouvoir en enlever un par la suite
  #pour ca on utilise np.split qui prend en argument:
  # 1-l'array qu'il faut diviser
  # 2-le nombre de sous-array a produire (donc on split en combien de partie), la je l'est mis a old_weights.shape[shape_axis] car on doit splitter en un nb d'elements
  # egal aux output channels (si c'est le layer ou on supprime le filtre) ou egal ou output channel si c'est le layer suivant
  # 3-l'axis concerné donc le dernier si c'est les output channels (layer ou on supprime le filtre) et l'avant dernier pour les input channels (layer qui suit celui pour
  # lequel on supprime les filtres) de facon general c'est len(old_weights.shape)+shape_axis = 4 +shape_axis (on fait plus car shape axis est negatif) donc ça 
  # nous donne 4-1=3 pour les output channels ou bien 4-2=2 pour les input channels
  
  split_weights=np.split(old_weights, old_weights.shape[shape_axis], axis=len(old_weights.shape)+shape_axis)
  split_weights=np.array(split_weights)
  
  #maintenant dans split_weights on a les filtres dans des array séparés selon les input/output channels
  #maintenat on peut supprimer les filtres qu'on veut du layer, les indice des filtres à supprimer sont dans list_filters_to_remove
  split_weights= np.delete(split_weights, list_filters_to_remove, axis=0)
  split_weights= split_weights.tolist()
  
  
  #une fois les filtres à élaguer supprimer on concatène les filtres restant qui constitue le layer après élagage
  #on utilise np.concatenate a qui on donne en argument split_weights qui est la liste des poids a concatener

  new_layer_weights = [np.concatenate(split_weights,
                                      axis=len(old_weights.shape)+shape_axis)] #on concatene les anciens poids avec les nouveaux neurones crees et on obtient new_layer_weights
                                                                               #on fait plus shape axis parce qu'il est négatif
  
  
  common_kwargs = {
      'name': old_module.name,
      'activation': old_module.activation,
      'use_bias':old_module.use_bias
  }
  for r_name in ('kernel_regularizer', 'bias_regularizer',
                 'activity_regularizer'):
    regularizer = getattr(old_module, r_name)
    if regularizer is not None:
      common_kwargs[r_name] = regularizer
  n_out_new = new_layer_weights[0].shape[-1] #on recupere le shape/dim des poids/matrice poids du layer apres growth
  
  
  #si on a un conv layer on cree un convlayer avec n_out_new le nouveau shape, weights les anciens poids et ceux ajoutes
  #on garde le meme stride, padding, kernel size
  if isinstance(old_module, tf.keras.layers.Conv2D):
    new_module = tf.keras.layers.Conv2D(
        n_out_new,
        kernel_size=old_module.kernel_size,
        strides=old_module.strides,
        padding=old_module.padding,
        weights=new_layer_weights,
        **common_kwargs)
    
    

  else:
    raise ValueError(f'Unexpected module: {old_module}')

  return new_module

#les 3 prochaines methodes (grow_new_ln_layer, grow_new_bn_layer, grow_new_dw_layer) sont utilisees dans la methode (un peu plus haut) add_neurons_identity
#donc c'est pour ajouter des identity neurons pour different type de layer

def grow_new_ln_layer(old_module, n_to_remove):
  """Grows a new identity LayerNormalization layer."""
  new_ln_weights = []
  # One for gamma, beta #je pense les arguments de la layer normalization (gamma et beta)
  for i in range(2):
    old_w = old_module.get_weights()[i] #on recupere les poids des neurones existants
    if i == 0:  # gamma
      new_w = np.ones([n_to_remove])
    else:  # beta
      new_w = np.zeros([n_to_remove])
    w = np.concatenate((old_w, new_w), axis=0) #on concatene les param existants et ceux ajoutes
    new_ln_weights.append(w)
  common_kwargs = {
      'epsilon': old_module.epsilon
  }
  for r_name in ('gamma_regularizer', 'beta_regularizer'):
    regularizer = getattr(old_module, r_name)
    if regularizer is not None:
      common_kwargs[r_name] = regularizer
  return tf.keras.layers.LayerNormalization(weights=new_ln_weights,
                                            **common_kwargs) #on cree un nouveau layer resultat du growth avec les anciens et nouveau poids

#meme fonctionnement que la methode precedente
def grow_new_bn_layer(old_module, n_to_remove):
  """Grows a new identity BatchNormalization layer."""
  new_bn_weights = []
  # One for gamma, beta, moving_mean and moving_variance
  for i in range(4):
    old_w = old_module.get_weights()[i]
    if i in (1, 2):  # beta, moving_mean
      new_w = np.zeros([n_to_remove])
    else:  # gamma, moving variance
      new_w = np.ones([n_to_remove])
    w = np.concatenate((old_w, new_w), axis=0)
    new_bn_weights.append(w)
  common_kwargs = {
      'epsilon': old_module.epsilon
  }
  for r_name in ('gamma_regularizer', 'beta_regularizer'):
    regularizer = getattr(old_module, r_name)
    if regularizer is not None:
      common_kwargs[r_name] = regularizer
  return tf.keras.layers.BatchNormalization(weights=new_bn_weights,
                                            **common_kwargs)


#dansl'article ils expliquent que quand on fait un growing des depthwise conv layer on initilise la depthwise comme etant l'identite, c'est pour ca que cette methode
#est definir
def grow_new_dw_layer(old_module, n_to_remove):
  """Adds identity neurosn to the depthwise convolutional layers."""
  old_weights = old_module.get_weights()[0] #on recupere les poids existants
  shape_new = list(old_weights.shape) #cette ligne et la suivante permettent de definir la dimension du layer apres le growth (ajout d'un nombre de n_to_remove neurone)
  shape_new[-2] = n_to_remove
  new_weights = np.zeros(shape_new, dtype=old_weights.dtype) #variable qui stocke les nouveaux poids
  mid_index_x = new_weights.shape[0] // 2
  mid_index_y = new_weights.shape[1] // 2
  new_weights[mid_index_x, mid_index_y, Ellipsis] = 1.
  new_layer_weights = [np.concatenate((old_weights, new_weights),
                                      axis=-2)] #on concatene les anciens/nouveaux poids

  # Assuming bias is the second weight.
  if old_module.use_bias:
    new_bias = old_module.get_weights()[1]
    new_neuron_bias = np.zeros([n_to_remove])
    new_bias = np.concatenate((new_bias, new_neuron_bias), axis=0)
    new_layer_weights.append(new_bias)

  regularizer_kwargs = {}
  for r_name in ('kernel_regularizer', 'bias_regularizer',
                 'activity_regularizer'):
    regularizer = getattr(old_module, r_name)
    if regularizer is not None:
      regularizer_kwargs[r_name] = regularizer
  #on cree un nouveau depthwise conv layer qui est le resultat du growth
  new_module = tf.keras.layers.DepthwiseConv2D(
      kernel_size=old_module.kernel_size,
      name=old_module.name,
      activation=old_module.activation,
      use_bias=old_module.use_bias,
      strides=old_module.strides,
      padding=old_module.padding,
      weights=new_layer_weights,
      **regularizer_kwargs)
  return new_module

#fonction utilisee dans les fonctions precedente
def norm_l2(tensor, keep_dim):
  norm_axes = list(range(len(tensor.shape)))
  del norm_axes[keep_dim]
  return tf.sqrt(tf.reduce_sum(tf.pow(tensor, 2), axis=norm_axes))

#fonction utilisee dans les fonctions precedente
def normalize_l2(tensor, axis):
  assert axis in (-2, -1)
  norm = norm_l2(tensor, axis)
  scale_recipe = '...ij,i->...ij' if (axis == -2) else '...ij,j->...ij'
  return tf.einsum(scale_recipe, tensor, 1 / norm)
