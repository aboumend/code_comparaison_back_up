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
SUPPORTED_LAYERS = (tf.keras.layers.Dense, tf.keras.layers.Conv2D)


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

  def add_neurons(self, n_new, new_weights='zeros', scale=1.,
                  is_outgoing=False, scale_method='mean_norm',
                  new_bias='zeros'):
    #voir leur explication
    """Adds new neurons and creates a new layer.

    New weights are scaled (if not zero) to have l2-norm equal to the mean
    l2-norm of the existing weights.
    TODO Unify splitting and adding neurons.
    Args:
      n_new: number of neurons to add.
      new_weights: 'zeros', 'random' or np.ndarray.
      scale: float, scales the new_weights multiplied with the mean norm of
        the existing weights.
      is_outgoing: bool, if true adds outgoing connections from the new neurons
        coming from previous layers. In other words number of neurons in current
        layer stays constant, but they aggregate information from n_new many
        new neurons.
      scale_method: str, Type of scaling to be used when initializing new
        neurons.
        - `mean_norm` means they are normalized using the mean norm of
        existing weights.
        - `fixed` means the weights are multiplied with scale directly.
      new_bias: str, 'zeros' or 'ones'.
    """
    old_module = self.layer
    #le assert teste si un boolean est a true: si c'est true rien ne se passe sinon on a une exception
    assert old_module.built
    assert new_bias in ('zeros', 'ones')
    assert isinstance(old_module, SUPPORTED_LAYERS) #verifier si c'est un fc layer ou un conv layer
    self.layer = grow_new_layer(old_module, n_new, new_weights, scale,
                                is_outgoing=is_outgoing, new_bias=new_bias,
                                scale_method=scale_method) #appel de la fonction qui fait le travail du growing
    #on modifie self.layer donc on remplace le layer existant (oldmodule) par un autre layer a qui on a ajoute des neurones (a subit le grow) 

  #pour l'utilite de add_neurons_identity dans le fichier growers.py ils expliquent que s'il y a des layer de nromalization/depthwiseconv entre deux layer concernes par le growth
  # (les layers concernes sont le layers qui subit le growth et le layers suivant car recoit les nouveaux poids), les layers entre les deux concernes
  #(notamment ceux de la normalization/depthwiseconv) on leur fait un identity growing
  #quand on dit identite je pense c'est au sens (0,1)
  def add_neurons_identity(self, n_new):
    """Adds identity neurons for various layer types.

    Args:
      n_new: number of neurons to add.
    """
    old_module = self.layer
    assert old_module.built
    
    #comme on rajoute des identity neurones, ils retrouvent d'abord quel est le type du layer (en utilisant isinstance(oldmodule, type layer))
    #puis ils appellent une fonction qui fait le grow selon le type du layer en passant n_new (nb neurones a ajouter) en argument
    if isinstance(old_module, tf.keras.layers.BatchNormalization):
      self.layer = grow_new_bn_layer(old_module, n_new)
    elif isinstance(old_module, tf.keras.layers.LayerNormalization): #normalization differente de la batch norm voir la doc si necessaire
      self.layer = grow_new_ln_layer(old_module, n_new)
    elif isinstance(old_module, tf.keras.layers.DepthwiseConv2D): #cas du mobilenet, ils definissent l'identite parce que dans l'article ils expliquent que lors du
                                                                  #grow d'un depthwise convlayer, la depthwise conv est initialise comme etant l'identite
      self.layer = grow_new_dw_layer(old_module, n_new)
    else:
      raise ValueError(f'layer: {old_module} of {type(old_module)} is not '
                       'supported.') #si oldmodule ne fait pas partie des layer supporte donc conv et fclayer


def grow_new_layer(old_module, n_new, new_weights, scale, is_outgoing=False,
                   scale_method='mean_norm', new_bias='zeros'):
  #ca sert a creer un nouveau layer mais pas au sens on a ajoute un nouveau, c'est plutot creer un nouveau layer auquel on a ajoute les neurones
  #donc ca sert a remplacer un certain layer par ce certain layer auquel on ajoute les nouveaux neurones
  """Creates new layer after adding incoming our outgoing connections.

  Args:
    old_module: Old layer to grow from. One of layers.SUPPORTED_LAYERS.
    n_new: number of neurons to add.
    new_weights: 'zeros', 'random' or np.ndarray.
    scale: float, scales the new_weights multiplied with the mean norm of
      the existing weights.
    is_outgoing: bool, True if the outgoing connections of the new neurons are
      being added to the next layer. In this case, no new neurons are generated;
      instead existing neurons receive new incoming connections.
    scale_method: str, Type of scaling to be used when initializing new
      neurons.
      - `mean_norm` means they are normalized using the mean norm of
      existing weights.
      - `fixed` means the weights are multiplied with scale directly.
    new_bias: str, zeros or ones.
  Returns:
    layer of same type as the old_module.
  """
  old_weights = old_module.get_weights()[0] #on prend les poids de oldmodule (layer avant growth)
  shape_axis = -2 if is_outgoing else -1

  if scale_method == 'mean_norm': #scales the new_weights multiplied with the mean norm of the existing weights
    magnitude_new = np.mean(norm_l2(old_weights, keep_dim=shape_axis).numpy())
    magnitude_new *= scale #argument de la methode
  elif scale_method == 'fixed':
    # We don't use the scale of existing weights for initialization.
    magnitude_new = scale
  else:
    raise ValueError(f'Not supported scale_method, {scale_method}')

  shape_new = list(old_weights.shape)
  shape_new[shape_axis] = n_new #ca doit etre pour changer les dimensions et ajouter les nouveaux neurones 

  if isinstance(new_weights, np.ndarray): #donc la les new weights sont donne dans un nd array
    assert new_weights.shape == tuple(shape_new)
    # Normalize to unit norm and then scale.
    #on fait un scale de ces new weights et on obtient les new_neurons
    normalized_w = normalize_l2(new_weights, axis=shape_axis).numpy()
    new_neurons = normalized_w * magnitude_new
  elif new_weights == 'random':#les new_weights sont random
    normalized_w = normalize_l2(np.random.uniform(size=shape_new),
                                axis=shape_axis).numpy()
    # Normalize to unit norm and then scale.
    #on genere des nombre alea pour les poids puis on fait un scale et on cree new_neurons
    new_neurons = normalized_w * magnitude_new
  elif new_weights == 'zeros': #si les nouveaux poids sont a zero on les met juste a zero
    new_neurons = np.zeros(shape_new)
  else:
    raise ValueError('new_weights: %s is not valid' % new_weights)
  new_layer_weights = [np.concatenate((old_weights, new_neurons),
                                      axis=shape_axis)] #on concatene les anciens poids avec les nouveaux neurones crees et on obtient new_layer_weights

  # Assuming bias is the second weight.
  #si le neurone contient un bias je pense qu'on doit aussi le creer/initialiser/modifier (peut etre juste modifier comme on a un bias per layer)
  if old_module.use_bias:
    new_bias_weights = old_module.get_weights()[1] #on recupere le bias du layer avant growth (old_module) (on a un bias per layer)
    if not is_outgoing:
      new_neuron_bias = (np.zeros([n_new]) if (new_bias == 'zeros') else
                         np.ones([n_new]))
      new_bias_weights = np.concatenate((new_bias_weights, new_neuron_bias),
                                        axis=0)
    new_layer_weights.append(new_bias_weights) #on rajoute le bias a la liste des poids du layer apres growth

  common_kwargs = {
      'name': old_module.name,
      'activation': old_module.activation,
      'use_bias': old_module.use_bias
  }
  for r_name in ('kernel_regularizer', 'bias_regularizer',
                 'activity_regularizer'):
    regularizer = getattr(old_module, r_name)
    if regularizer is not None:
      common_kwargs[r_name] = regularizer
  n_out_new = new_layer_weights[0].shape[-1] #on recupere le shape/dim des poids/matrice poids du layer apres growth
  
  #si on a un dence layer: on cree un dence/fclayer avec: n_out_new: nb neurones (qu'on a recupere dans l'instru precedente) et
  #les valeurs des poids des anciens poids et ceux ajoutes (new_layer_weights)
  if isinstance(old_module, tf.keras.layers.Dense):
    new_module = tf.keras.layers.Dense(
        n_out_new,
        weights=new_layer_weights,
        **common_kwargs)

  #si on a un conv layer on cree un convlayer avec n_out_new le nouveau shape, weights les anciens poids et ceux ajoutes
  #on garde le meme stride, padding, kernel size
  elif isinstance(old_module, tf.keras.layers.Conv2D):
    new_module = tf.keras.layers.Conv2D(
        n_out_new,
        kernel_size=old_module.kernel_size,
        strides=old_module.strides,
        padding=old_module.padding,
        weights=new_layer_weights,
        **common_kwargs)

  #====> en resume, pour le layer sur lequel on applique le grow, on recupre les poids des neurones existants (pour des calculs pour le scaling), puis on 
  #      initialise les nouveaux poids selon une certaine facon (zero, random ou via un ndarray), on applique le scling dessus et on conctactene les anciens poids 
  #      et les nouveaux. On cree un nouveau layer avec les anciens et nouveau poids (ce nouveau layer est le resultat du growth d'un layer existant), ca peut etre un
  #      fc layer ou un conv-layer.

  else:
    raise ValueError(f'Unexpected module: {old_module}')

  return new_module

#les 3 prochaines methodes (grow_new_ln_layer, grow_new_bn_layer, grow_new_dw_layer) sont utilisees dans la methode (un peu plus haut) add_neurons_identity
#donc c'est pour ajouter des identity neurons pour different type de layer

def grow_new_ln_layer(old_module, n_new):
  """Grows a new identity LayerNormalization layer."""
  new_ln_weights = []
  # One for gamma, beta #je pense les arguments de la layer normalization (gamma et beta)
  for i in range(2):
    old_w = old_module.get_weights()[i] #on recupere les poids des neurones existants
    if i == 0:  # gamma
      new_w = np.ones([n_new])
    else:  # beta
      new_w = np.zeros([n_new])
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
def grow_new_bn_layer(old_module, n_new):
  """Grows a new identity BatchNormalization layer."""
  new_bn_weights = []
  # One for gamma, beta, moving_mean and moving_variance
  for i in range(4):
    old_w = old_module.get_weights()[i]
    if i in (1, 2):  # beta, moving_mean
      new_w = np.zeros([n_new])
    else:  # gamma, moving variance
      new_w = np.ones([n_new])
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
def grow_new_dw_layer(old_module, n_new):
  """Adds identity neurosn to the depthwise convolutional layers."""
  old_weights = old_module.get_weights()[0] #on recupere les poids existants
  shape_new = list(old_weights.shape) #cette ligne et la suivante permettent de definir la dimension du layer apres le growth (ajout d'un nombre de n_new neurone)
  shape_new[-2] = n_new
  new_weights = np.zeros(shape_new, dtype=old_weights.dtype) #variable qui stocke les nouveaux poids
  mid_index_x = new_weights.shape[0] // 2
  mid_index_y = new_weights.shape[1] // 2
  new_weights[mid_index_x, mid_index_y, Ellipsis] = 1.
  new_layer_weights = [np.concatenate((old_weights, new_weights),
                                      axis=-2)] #on concatene les anciens/nouveaux poids

  # Assuming bias is the second weight.
  if old_module.use_bias:
    new_bias = old_module.get_weights()[1]
    new_neuron_bias = np.zeros([n_new])
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
