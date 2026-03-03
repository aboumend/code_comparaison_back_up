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
from scipy.sparse.linalg.eigen import arpack #package which provides routines for quickly finding a few eigenvalues/eigenvectors of large sparse matrices
import tensorflow as tf


class LayerGrower():
  """Base class for growing layer algorithms.

  Subclasses should implement grow_neurons.
    grad_fn: Should return list of variables and return aggregated gradients.
    grow_layers: list of GrowLayers. There are often 2 layers. First is the
      one we are adding neurons and the second is the layer that consumes
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
  epsilon = 0.
  scale_method = 'mean_norm'
  strategy = None
  compile_fn = lambda: None #parce que dans tensorflow on fait un model.compile(ou on donne en argument le loss, optimizer, metrics evaluation) avant le model.fit
  loss_fn = lambda x: x

  def grow_neurons(self, grow_layers, batch_data, **kwargs): #sera implemente dans les sous-classes, la c'est juste la classe de base
    raise NotImplementedError()

## Random growing
class AddRandom(LayerGrower): #herite de layergrower, doit implementer grow_neurons en faisant un random growing
  """Implements random growing."""
  is_outgoing_zero = False #si c'est les poids sortants/ je pense on le met a true ou a false selon est-ce que les neurones ajoutes ont des poids entrant ou sortant dans le layer considere
  is_all_zero = False #si les poids entrants et sortants sont a zero (je pense)

  def grow_neurons(self, grow_layers, batch_data, n_new=1, scale=1.):
    del batch_data #supprime l'objet batch_data car pour le random grow on n'a pas besoin du dataset pour des calculs qu'impose l'approche ici c'est random
    scales = (self.epsilon, scale)
    new_bias = 'zeros' #donc pour un growing random on met les nouveaux bias a zero
    if self.is_all_zero: #poids entrants et sortants a zero
      scales = (self.epsilon, self.epsilon)
      new_bias = 'ones'
    elif self.is_outgoing_zero: #poids sortant a zero
      scales = (scale, self.epsilon)
    for i, layer in enumerate(grow_layers): 
      #pour ce que veut dire grow_layers voir le comment de la classe LayerGrower(en gros c'est la liste des layer concernes par le growth)
      if i == 0:
        # First layer: donc normalement celui auquel on rajoute les nouveaux neurones
        layer.add_neurons(n_new, new_weights='random', is_outgoing=False,
                          scale=scales[0], scale_method=self.scale_method,
                          new_bias=new_bias) #add_neurons est definie dans le fichier layers.py, c'est la methode qui englobe le processus d'ajout de neurone a un layer
                                            #en gros ca permet de creer le layer resultat du growth (layer avant growth + nouveau neurones)
      elif i == (len(grow_layers) - 1):
        # Last layer: layer qui consomme/recoit des neurones de first layer
        layer.add_neurons(n_new, new_weights='random', is_outgoing=True,
                          scale=scales[1], scale_method=self.scale_method)
      else: #layer de normalization/depthwiseconv qui sont entre ce qu'on a appele first et last layer, on leur ajoute des identity neurones selon la methode
            #add_neurons_identity du fichier layers.py
        if isinstance(layer, glayers.GrowLayer):
          layer.add_neurons_identity(n_new)


class AddFirefly(AddRandom): #growing selon firefly, sous-classe de AddRandom mais ca initialise les poids de facon a reduire directement le loss je pense
  #je pense que c'est une version simplifie par rapport a l'article firefly
  """Implements Firefly style growing using direct optimization.

  Implements Eq:4 from the paper without extra candidates and splitting.
  https://arxiv.org/abs/2102.08574
  """
  optim_n_step = 100
  optim_fn = lambda self: tf.keras.optimizers.Adam()

  def grow_neurons(self, grow_layers, batch_data, n_new=1, scale=1.):
    n_old_neuron = grow_layers[0].weights[0].shape[-1] #nb neurones existants (nb neurones avant le growth)
    # First add neurons randomly (selon l'algo firefly)
    super().grow_neurons(grow_layers, batch_data, n_new=n_new, scale=scale) #donc fait un random growing en appelant le grow_neurons de la classe mere (AddRandom)
    self.compile_fn() 
    # Now optimize the random initialization (selon l'algo firefly)
    layer_tuple = grow_layers[0], grow_layers[-1]

    optimizer = self.optim_fn() #definie juste avant cette method, en gros c'est l'optimizer utilise (ex:adam)
    target_magnitudes = []
    # Record the magnitude of the new_weights.
    for concat_axis, layer in zip([-1, -2], layer_tuple):
      _, new_weights = tf.split(layer.weights[0], [n_old_neuron, -1],
                                axis=concat_axis)
      target_magnitudes.append(
          np.mean(glayers.norm_l2(new_weights, keep_dim=concat_axis)))
    logging.info('Minimizing loss.')
    weights = [l.weights[0] for l in layer_tuple]

    @tf.function
    def update_fn(inputs):
      with tf.GradientTape() as tape:
        loss = self.loss_fn(inputs)
      grads = tape.gradient(loss, weights)
      masked_grads = []
      for concat_axis, grad in zip([-1, -2], grads):
        # Apply gradient only on new weights, zero out the rest.
        old_wgrad, new_wgrad = tf.split(grad, [n_old_neuron, -1],
                                        axis=concat_axis)
        masked_grad = tf.concat([tf.zeros_like(old_wgrad), new_wgrad],
                                axis=concat_axis)
        masked_grads.append(masked_grad)
      optimizer.apply_gradients(zip(masked_grads, weights))
      # Project new weights back to the target magnitude.
      return loss

    log_freq = self.optim_n_step // 10
    for i in range(self.optim_n_step):
      per_replica_losses = self.strategy.run(update_fn, args=(batch_data,))
      loss = self.strategy.reduce(
          tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
      if i % log_freq == 0:
        logging.info('Firefly iter: %d, loss: %s', i, loss)
      for concat_axis, weight, target_magnitude in zip(
          [-1, -2], weights, target_magnitudes):
        old_w, new_w = tf.split(weight, [n_old_neuron, -1],
                                axis=concat_axis)
        normalized_w = glayers.normalize_l2(new_w, axis=concat_axis)
        normalized_new_w = normalized_w * target_magnitude
        weight.assign(
            tf.concat([old_w, normalized_new_w], axis=concat_axis))
    logging.info('Firefly final loss: %s', loss.numpy())


class AddGradmaxOptim(AddRandom):#implemente gradmaxopt (donc methode sans svd mais avec optim direct avec gradient descent)
  """Implements Gradmax using direct optimization."""
  optim_n_step = 100
  optim_fn = lambda self: tf.keras.optimizers.Adam()

  def grow_neurons(self, grow_layers, batch_data, n_new=1, scale=1.):
    # For simplicity we do full backward and forward pass here, but note that
    # only thing we need here is inputs at l-1 and gradients at l+1. Those stay
    # same and don't need to be re-calculated each time.
    n_old_neuron = grow_layers[0].weights[0].shape[-1] #nb neurones avant growth
    # First add neurons randomly
    super().grow_neurons(grow_layers, batch_data, n_new=n_new, scale=scale) #ajouter des neurones random en appelant le grow_neurons de la classe mere
    self.compile_fn()
    # Now optimize the random initialization
    if self.is_outgoing_zero: #poids sortants a zero, on optimise les entrants
      # We optimize incoming weights
      optim_layer, grad_layer = grow_layers[0], grow_layers[-1] #on recupere les layers a optimiser depuis la liste grow_layers: layer concernes par le growth (voir les comments de LayerGrower)
      concat_axis = -1 #axes sur lequel on concatene je pense
      grad_slic_fn = lambda a: a[Ellipsis, n_old_neuron:, :] 
    else:
      # We optimize outgoing weights
      optim_layer, grad_layer = grow_layers[-1], grow_layers[0]
      concat_axis = -2
      grad_slic_fn = lambda a: a[Ellipsis, n_old_neuron:] #ellipsis c'est pour le slicing donc la je pense que la lambda function fais du slicing

    optimizer = self.optim_fn() #c'est l'optimizer adam on l'a definit juste avant la methode grow_neurons
    target_magnitude = None
    # Record the magnitude of the new_weights.
    _, new_weights = tf.split(optim_layer.weights[0], [n_old_neuron, -1],
                              axis=concat_axis) #divise le tensor en subtensors (le tensor en question c'est optim_layer.weights)
    target_magnitude = np.mean(glayers.norm_l2(new_weights, #glayers c'est le fichier layers.py on l'a importe avec ce nom
                                               keep_dim=concat_axis))
    logging.info('Target magnitude: %s', target_magnitude)
    optim_layer_weight = optim_layer.weights[0]
    logging.info('Minimizing loss.')

    @tf.function
    def update_fn(inputs):
      with tf.GradientTape(persistent=True) as tape: #on utilise le tape car ca stocke toutes les operations faites ce qui permet de calculer les gradients par la suite
        loss = self.loss_fn(inputs) #calcul loss, les inputs je pense que c'est le mini batch du dataset qu'utilise gradmax 
        grad_layer_weight = grad_layer.weights[0] #le gradlayer c'est le layer qui consomme les neurones du layer qui subut le growth
        inner_grad = tape.gradient(loss, grad_layer_weight) #on calcule le gradient du loss par rapport aux poids de grad_layer
        # Maximize gradient norm.
        final_loss = -tf.norm(grad_slic_fn(inner_grad)) #le grad_slic_fn est definie comme une lambda function et normalement ca fait du slicing faire un affichage pour comprendre
      grad = tape.gradient(final_loss, optim_layer_weight) #on calcule le gradient de final_loss par rapport a optim_layer_weight
      # Apply gradient only on new weights, zero out the rest.
      old_wgrad, new_wgrad = tf.split(grad, [n_old_neuron, -1],
                                      axis=concat_axis) #on separe les gradients des anciens poids de ceux des nouveaux
      masked_grad = tf.concat([tf.zeros_like(old_wgrad), new_wgrad],
                              axis=concat_axis) #on concatene les gradients des nouveau poids a des zero (selon leur comment on garde juste les gradients ds nouveux poids)
                                                #c'est comme s on met ceux des anciens a zero
      optimizer.apply_gradients([(masked_grad, optim_layer_weight)]) #step de l'optimizer applique seulement sur les nouveaux gradients
      return final_loss

    log_freq = self.optim_n_step // 10
    for i in range(self.optim_n_step): #le optim_n_step est ici fixe a 100
      per_replica_losses = self.strategy.run(update_fn, args=(batch_data,)) #on applique le update_fn qu'on vient de definir sur (je pense) le minibatch du dataset comme explique dans l'artcile on applique la methode sur un mini-batch
      loss = self.strategy.reduce(
          tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
      if i % log_freq == 0:
        logging.info('Gradmax-opt: %d, loss: %s', i, loss)
      # Project new weights back to the target magnitude.
      # Record the magnitude of the new_weights.
      old_w, new_w = tf.split(optim_layer_weight, [n_old_neuron, -1],
                              axis=concat_axis)
      normalized_w = glayers.normalize_l2(new_w, axis=concat_axis)
      normalized_new_w = normalized_w * target_magnitude
      optim_layer_weight.assign(
          tf.concat([old_w, normalized_new_w], axis=concat_axis))
    logging.info('Grad-max-opt final loss: %s', loss.numpy())

  def _grow_neurons_legacy(self, grow_layers, batch_data, n_new=1, scale=1.):
    """Old function to calculate gradmax-opt initialization efficiently."""
    # Note that this version doesn't work currently.
    # The issue here is the inputs are shared, thus it is a challenge to
    # uptained them and reshard. This path is efficient but might be
    # unncessearily complicated, thus we do full pass like above.
    logging.warning('This function is not doing the right thing in multi-worker'
                    'setting.')
    n_old_neuron = grow_layers[0].weights[0].shape[-1]
    # First get the output gradient at l+1 and the input at l-1.
    aux_tensor = []
    # For simplicity we do full backward and forward pass here, but note that
    # only thing we need here is inputs at l-1 and gradients at l+1. Those stay
    # same and don't need to be re-calculated each time.
    def next_layer_callback(next_inputs, next_outputs):
      aux_tensor.append(tf.zeros_like(next_outputs))
      return next_inputs, (next_outputs + aux_tensor[-1])
    grow_layers[-1].add_callback('add_zeros', next_layer_callback)
    inp_tensor = []
    def first_layer_callback(next_inputs, next_outputs):
      inp_tensor.append(next_inputs)
      return next_inputs, next_outputs
    grow_layers[0].add_callback('collect_inp', first_layer_callback)

    def grad_fn(inputs):
      with tf.GradientTape() as tape:
        loss = self.loss_fn(inputs)
      return tape.gradient(loss, aux_tensor[0])
    per_replica_grads = self.strategy.run(grad_fn, args=(batch_data,))
    out_grads = self.strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_grads, axis=None)

    # Second add neurons randomly
    super().grow_neurons(grow_layers, batch_data, n_new=n_new,
                         scale=scale)
    self.compile_fn()
    # Now optimize the random initialization
    if self.is_outgoing_zero:
      # We optimize incoming weights
      optim_layer, grad_layer = grow_layers[0], grow_layers[-1]
      concat_axis = -1
      grad_slic_fn = lambda a: a[Ellipsis, n_old_neuron:, :]
    else:
      # We optimize outgoing weights
      optim_layer, grad_layer = grow_layers[-1], grow_layers[0]
      concat_axis = -2
      grad_slic_fn = lambda a: a[Ellipsis, n_old_neuron:]

    optimizer = self.optim_fn()
    target_magnitude = None
    # Record the magnitude of the new_weights.
    _, new_weights = tf.split(optim_layer.weights[0], [n_old_neuron, -1],
                              axis=concat_axis)
    target_magnitude = np.mean(glayers.norm_l2(new_weights,
                                               keep_dim=concat_axis))
    logging.info('Target magnitude: %s', target_magnitude)
    optim_layer_weight = optim_layer.weights[0]

    @tf.function
    def update_fn(inp_tensor, out_grads):
      with tf.GradientTape(persistent=True) as tape:
        x = inp_tensor
        for l in grow_layers:
          x = l(x, training=True)
        # This simulates having output grads at the end. But it is way more
        # efficient as we don't need to run the input again through the whole
        # network.
        # dL/dx = out_grads because grad_x(x * y) = y
        loss = tf.reduce_sum(x*out_grads)
        grad_layer_weight = grad_layer.weights[0]
        inner_grad = tape.gradient(loss, grad_layer_weight)
        # Maximize gradient norm.
        final_loss = -tf.norm(grad_slic_fn(inner_grad))
      grad = tape.gradient(final_loss, optim_layer_weight)
      # Apply gradient only on new weights, zero out the rest.
      old_wgrad, new_wgrad = tf.split(grad, [n_old_neuron, -1],
                                      axis=concat_axis)
      masked_grad = tf.concat([tf.zeros_like(old_wgrad), new_wgrad],
                              axis=concat_axis)
      optimizer.apply_gradients([(masked_grad, optim_layer_weight)])
      return final_loss
    logging.info('Maximizing gradients')
    log_freq = self.optim_n_step // 10
    for i in range(self.optim_n_step):
      per_replica_losses = self.strategy.run(update_fn,
                                             args=(inp_tensor[0], out_grads))
      loss = self.strategy.reduce(
          tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
      if i % log_freq == 0:
        logging.info('GradmaxOptim iter: %d, loss: %s', i, loss.numpy())
      # Project new weights back to the target magnitude.
      # Record the magnitude of the new_weights.
      old_w, new_w = tf.split(optim_layer_weight, [n_old_neuron, -1],
                              axis=concat_axis)
      normalized_w = glayers.normalize_l2(new_w, axis=concat_axis)
      normalized_new_w = normalized_w * target_magnitude
      optim_layer_weight.assign(
          tf.concat([old_w, normalized_new_w], axis=concat_axis))
    logging.info('Final Grad-Norm: %f', -loss.numpy())


class AddGradmax(LayerGrower):
  """Implements Gradmax using auxiliary layer formulation."""

  def grow_neurons(self, grow_layers, batch_data, n_new=1, scale=1.):
    if len(grow_layers) == 2: #dans grow_layers on a les layers concernes par le growth, s'il y en a que 2, c'est le layer qui subit le growth et le layer qui consomme les poids du layer qui subit le growth
      current_layer, next_layer = grow_layers #pour separer les deux layers
      identity_layers = [] #vide car on n'a pas de layers de normalization ou depthwise conv entre le layer qui subit le growth et celui qui consomme ces poids
    else: #cas contraire on a des layer de normalization et depthwise conv entre le layer qui subit le grwoth et celui qui consomme ces poids
      assert len(grow_layers) > 2
      current_layer, next_layer = grow_layers[0], grow_layers[-1] #layer qui subit le growth et celui qui consomme ces poids
      identity_layers = grow_layers[1:-1] #layer normalization et depthwise conv situe entre les layers precedents
    # There is only one candidate
    growth_candidates = [(current_layer, next_layer)]
    if not isinstance(current_layer.layer, type(next_layer.layer)): 
      #cas ou le layer qui subit le growth et celui qui consomme ces poids (layer suivant) ne sont pas de meme type on fait un random grow
      
      # This is a temporary fix for dealing with heteregonous layers.
      # When two consecutive layers are different we grow randomly.
      # For example when a convolutional layer is followed by a fully connected
      # layer.
      logging.info('Growing randomly layers: %s %s, %s %s',
                   current_layer.layer.name, type(current_layer.layer),
                   next_layer.layer.name, type(next_layer.layer))
      tmp_grower = AddRandom() #comme on fait un random grow
      tmp_grower.epsilon = self.epsilon
      tmp_grower.scale_method = self.scale_method
      tmp_grower.is_outgoing_zero = False
      tmp_grower.grow_neurons(grow_layers, batch_data, n_new=n_new, scale=scale)
      return #donc si les layers ne sont pas de meme type on fait un random growth et on s'arrete la

    #SINON:
    unused_eigenvals, eigenvecs = self.get_growth_directions( #get_growth_directions: Efficiently retrieves eigen-decomposition for a set of candidates, voir la suite du code
        batch_data, growth_candidates, [n_new])[0]
    #get_growth_directions retourne le resultat de svd
    # Grow incoming connections
    current_layer.add_neurons(n_new, new_weights='random', is_outgoing=False,
                              scale=self.epsilon,
                              scale_method=self.scale_method) #add_neurons est defini dans le fichier layers.py et elle permet de creer le layer resultat du growth
    # Initialize intermediate layers as identity.
    for layer in identity_layers:
      if isinstance(layer, glayers.GrowLayer):
        layer.add_neurons_identity(n_new)#selon les fonction du fichier layers.py
    # Top-k Eigenvectors
    #ils initialisent new_weights selon les resultats retournes par la mthode get_growth_directions
    #meme s'ils note eigenvecs je pense qu'il s'agit des singular vectors
    new_weights = eigenvecs[:, :n_new]
    c_shape = next_layer.weights[0].shape
    if len(c_shape) == 4:
      # First reshape each neuron and then transpose last 2 dimensions.
      new_filter_shape = c_shape[:2] + [c_shape[-1], n_new]
      new_weights = np.reshape(new_weights, new_filter_shape)
      new_weights = np.transpose(new_weights, axes=(0, 1, 3, 2))
    elif len(c_shape) == 2:
      new_weights = new_weights.T
    #on obtient les new_weight selon eigenvecs et on appelle add_neurons du fichier layers.py pour creer le layer resultats du growth
    next_layer.add_neurons(n_new, new_weights=new_weights, scale=scale,
                           is_outgoing=True,
                           scale_method=self.scale_method)

  def get_growth_directions(self, batch_data, growth_candidates, n_grows): #fait le calcul des singular vectors
    """Efficiently retrieves eigen-decomposition for a set of candidates."""
    # Adding all callbacks.
    aux_layers = [] #je pense que c'est l'auxilary variables dont ils parlent dans l'annexe A
    post_process_fns = []
    for current_layer, next_layer in growth_candidates:
      aux_layer, post_process_fn = self.get_aux_layer(current_layer.layer,
                                                      next_layer.layer)
      post_process_fns.append(post_process_fn)
      def grow_layer_callback(inputs, outputs, aux_layer=aux_layer,
                              next_layer=next_layer):
        add_h = aux_layer(inputs)
        def next_layer_callback(next_inputs, next_outputs):
          return next_inputs, (next_outputs + add_h)
        next_layer.add_callback('add_aux', next_layer_callback)
        return inputs, outputs
      current_layer.add_callback('pass_aux', grow_layer_callback)
      aux_layers.append(aux_layer)

    def grad_fn(inputs):
      with tf.GradientTape() as tape:
        loss = self.loss_fn(inputs)
      grad_vars = [aux_layer.weights[0] for aux_layer in aux_layers]
      return tape.gradient(loss, grad_vars) #calcul du graident du loss par rapport a l'auxilary layer
    per_replica_grads = self.strategy.run(grad_fn,
                                          args=(batch_data,)) #pour lancer le calcul des gradinet par rapport a aux_layer je pense (utiliser strategy c'est pour distribuer le code sur des gpu ou autre)
    aux_grads = self.strategy.reduce(#je pense que le but du reduce est de convertir le resultat du run (qui est dans un format specifique) en tensor
        tf.distribute.ReduceOp.SUM, per_replica_grads, axis=None) #le tf.distribute.ReduceOp.SUM veut dire qu'on fait la somme des val obtenur dans per_replica_grads
    grow_matrices = [#avec ca on contruit la grow matrix qui est celle pour laquelle on applique svd
        post_process_fn(g)
        for g, post_process_fn in zip(aux_grads, post_process_fns)]#je pense post_process_fn calcule le gradient selon si c'est un conv ou fc layer (pas sur)
    # Reset Callbacks
    for current_layer, next_layer in growth_candidates:
      current_layer.reset_callbacks()
      next_layer.reset_callbacks()
    results = []
    # Calculate eigenvalues
    for grow_matrix, n_grow in zip(grow_matrices, n_grows):#calcul des val singuliere
      # M^{l+1} by M^{l+1}
      if n_grow > 0:
        # svds is equivalent to calling eigsh on M.T @ M (without materialiazing
        # this matrix) which is faster or slower (depending on the shape of M)

        _, s, vh = arpack.svds(grow_matrix, k=n_grow,
                               return_singular_vectors='vh')
        
        eigenvals, eigenvecs = (s**2)[::-1], vh[::-1].T
      else:
        s, _, v = tf.linalg.svd(grow_matrix)
        eigenvals, eigenvecs = s**2, v
      results.append((eigenvals, eigenvecs))
    return results

  def get_aux_layer(self, first_layer, second_layer):
    """Creates auxilarly layers for growing new neurons between layers."""
    l = tf.keras.layers
    if isinstance(first_layer, l.Dense) and isinstance(second_layer, l.Dense): #si on est entre deux dense layers on cree un dense layer
      aux_layer = l.Dense(second_layer.units, activation=None, use_bias=False,
                          kernel_initializer='zeros')
      post_process_fn = lambda a: a #identite
    elif (isinstance(first_layer, l.Conv2D) and
          isinstance(second_layer, l.Conv2D)):#si on est entre deux conv layers on cree un conv layer
      # Combined auxiliary kernel would be the size of k1+k2-1.
      kernel_size = [k1+k2-1 for k1, k2 in
                     zip(first_layer.kernel_size, second_layer.kernel_size)]
      # The auxiliary layer should have the combined stride.
      # Current implementation assumes tuple strides.
      strides = [(s1 + s2) if ((s1 > 1) and (s2 > 1)) else (s1 + s2 -1)
                 for s1, s2 in zip(first_layer.strides, second_layer.strides)]
      # Current implementation assumes paddings are same for the 2 layers.
      aux_layer = l.Conv2D(second_layer.filters, kernel_size, activation=None,
                           use_bias=False, padding=first_layer.padding,
                           kernel_initializer='zeros', strides=strides)
      post_process_fn = functools.partial(
          process_conv_aux_gradient, #voir methode suivante dans le code, ca permet de calculer le gradient des conv layer pour la growth matrix (donc celle pour laquelle on fait le svd je pense)
          second_kernel_size=second_layer.kernel_size)
    else:
      raise ValueError('Not Supported')

    return aux_layer, post_process_fn


def process_conv_aux_gradient(grad, second_kernel_size):
  """Process the gradients of convolutional layer to generate grow matrix.""" #je pense que ca rentre dans les details d'implementation pour les conv layer
  # shape(grad): ksize X ksize X m0 X m2 ; ksize=k1+k2-1
  # second_kernel_size == k2
  grad = tf.transpose(grad, perm=(2, 0, 1, 3))
  # shape(grad): m0 X ksize X ksize X m2
  patched_grow_matrix = extract_image_patches(grad, second_kernel_size)
  # shape(patched_grow_matrix): m0 X k1 X k1 X (m2 * k2 * k2)
  grow_matrix = tf.reshape(patched_grow_matrix,
                           [-1, patched_grow_matrix.shape[-1]])
  # shape(patched_grow_matrix): (m0 * k1 * k1) X (m2 * k2 * k2)
  return grow_matrix


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
