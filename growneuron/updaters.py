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

"""Implements controllers for updating networks.
"""
## ce fichier est un peu plus high level puisqu'il utilise les deux precedents
## c'est celui qui englobe un peu tous
## ca permet de definir quand on fait le grow (donc dire si l'epoch/iter courante est une etape de growing ou pas)
## il utilise un tuple de layer candidat au growing et ils implementent deux possibilites: 
## --> faire un growing en round robin entre les candidats (classe RoundRobin)
## --> faire un growing de tous les layers candidats (classe AllAtOnce)
## il fait aussi des modif sur l'optimizer pour ajouter des zero je pense ca permet de faire de la place au poids des nouveaux neurones
## le but je pense est de faire les changements necessaires sur l'optimizer pour continuer le training et l'optimization du loss apres le growth
## quand je dis optimizer je parle de sgd/adam...


import itertools #Fonctions creant des iterateurs pour boucler efficacement
#from growneuron import growers
#from growneuron import layers
import growers
import layers
import tensorflow as tf


def pad_zeros_to(tensor, new_shape):
  #remboure un tensor avec des zero pour que sont final shape soit new_shape (les zero sont ajoutes a la fin pour chaque dimension)
  #le new_shape est suppose etre plus grand que le shape actuel du tensor
  """Pads a tensor with zeros such that final shape is new_shape.

  It expects the new_shape to be larger than the tensor.shape.
  Zeros are added to the end of each dimension.
  Args:
    tensor: 1d, 2d, 3d tensor.
    new_shape: list of dimensions where len(new_shape) == len(tensor.shape) #car len doit donner le nb de dimension je pense et ce nb reste inchang meme si on change le shape
  Returns:
    new tensor of shape `new_shape`.
  """
  old_shape = tensor.shape
  #je pense le but des conditions est de determiner l'axis ou on remboure avec des zero
  if len(old_shape) == 1:
    # Batchnorm or bias.
    diff_shape = [new_shape[-1] - old_shape[-1]] #-1 indexe le dernier element du tuple shape
    concat_axis = -1
  else:
    if old_shape[-2] == new_shape[-2]:
      # Input features are same, padding at axis=-1.
      concat_axis = -1
    else:
      concat_axis = -2
    diff_shape = list(old_shape)
    diff_shape[concat_axis] = new_shape[concat_axis] - old_shape[concat_axis]
  return tf.concat([tensor, tf.zeros(diff_shape)], axis=concat_axis)


class Updater():
  ## certain des attributs sont fixe depuis le fichier depuis lequel on lance le training
  """Implements common methods.

  Updaters should be created under strategy scope or strategy should be passed
  directly.
  Attr:
    network_grower: growers.LayerGrower #donc la classe mere/globale du fichier growers.py ou les methodes de growing sont implementees
    #pour grow_layer_tuples ca doit etre les layers candidats au growing donc ceux qui vont subir le growth(on leur ajoute des neurones)
    grow_layer_tuples: list of lists<glayers.GrowLayer>, candidates to be
      grown together with their outgoing weights. #GrowLayer est defini dans le fichier layers.py ou on definit comment creer le layer resultant du growing
    loss_fn: fn, Used to calculate loss. This function should get inputs
      as input and return loss.
    compile_fn: fn, Called to compile the model.#parce que dans tensorflow on fait un model.compile(ou on donne en argument le loss, optimizer, metrics evaluation) avant le model.fit
    update_frequency: int, Number of iterations before neurons are added.
    n_grow: int, number of neurons to grow at each growth step.
    n_grow_fraction: float, must be positive. Used together with initial width
      of candidate layers to decide n_neurons to grow at each growth step for
      each candidate separately. This approach is helpful when predicting the
      final architecture from the start as number of neurons added are fixed at
      the beginning for each layer. #ca doit etre le taux de reduction utilise pour obtenir la seed architecture (ex ils ont reduit vgg de 1/4) je pense
    start_iteration: int, to start growing
    n_growth_steps: int, number of times the network is grown.
    scale: int, passed to the grower.grow_neurons #voir fichier growers.py
    carry_optimizer: bool, If true the running averages are carried to the new
      optimizer after the growth. Since variables are recreated after growth
      this is necessary. #je pense qu'ils font ca parce que a chaque grow le layer qui subit le growth est remplace par un nouveau donc je pense qu'ils passent le
                         #running average a l'optimizer pour ne pas perturber le gradient descent
  """

  def __init__(self, network_grower, grow_layer_tuples, loss_fn=lambda x: x,
               compile_fn=lambda: None, update_frequency=1, n_grow=1,
               n_grow_fraction=None, start_iteration=None, n_growth_steps=None,
               scale=1., carry_optimizer=True): #pour la signification des arguments voir leur comment c'est clair
    assert update_frequency > 0
    assert n_grow > 0
    self._update_frequency = update_frequency
    self._carry_optimizer = carry_optimizer
    self._n_grow = n_grow
    self._n_grow_fraction = n_grow_fraction
    self._scale = scale
    if start_iteration is None:
      start_iteration = update_frequency #si on ne definit pas la start_iteration (premiere iter ou on fait le grow), par defaut on commence le growing apres avoir fait
                                         #le nb d'iter qui separent deux growth (update_frequency) c'est logique
    self._start_iteration = start_iteration
    self.loss_fn = loss_fn #fonction calcul loss, on le definit dans le fichier main ou on lance le training je pense
    self.compile_fn = compile_fn #le model.compile de tensorflow
    self.strategy = tf.distribute.get_strategy() #tf.distibute.Strategy est une api qui permet de repartir le training sur plusieurs gpu/tpu/equipement
                                                 #get_strategy retourne la startegie par defaut (strategie utilise pour ditribuer le training)
    self.network_grower = self._prepare_grower(network_grower) #voir methode juste en bas
    self._n_growth_steps = n_growth_steps
    self._growth_counter = 0 #je pense ca compte le nb de fois ou on a fait un growth
    self._set_grow_layer_tuples(grow_layer_tuples)

  def _prepare_grower(self, grower):
    if grower:
      grower.loss_fn = self.loss_fn
      grower.compile_fn = self.compile_fn
      grower.strategy = self.strategy
    return grower

  def copy_optimizer_slots(self, optimizer, old_variables, new_variables):
    """Copy old slots and pad with zeros for new neurons."""
    #on utilise la fonction pad_zeros_to definit en haut du code pour rembourer avec des zero, je pense les zero ajoutes font de la place pour les nouveau neurones
    for old_var, new_var in zip(old_variables, new_variables):
      for s_name in sorted(optimizer.get_slot_names()):
        old_slot_var = optimizer.get_slot(old_var, s_name)
        new_slot_var = optimizer.get_slot(new_var, s_name)
        # This is used to retrieve the part of the new slot used for the
        # old variables. This assumes new variables are appended to the end.
        new_slot_values = pad_zeros_to(old_slot_var, new_slot_var.shape)
        new_slot_var.assign(new_slot_values)

  def delete_optimizer_slots(self, optimizer, variables):
    """Deleted old variable slots from the optimizer."""
    for old_var in variables:
      key = (old_var._shared_name if old_var._in_graph_mode
             else old_var._unique_id)
      optimizer._slots.pop(key, None)

  def _set_grow_layer_tuples(self, grow_layer_tuples): #selon leur explications, le grow_layers_tuples c'est les candidates to be grown (donc layer qui subiront le grow je suppose)
    """Sets the tuple of layers for growing."""
    if not grow_layer_tuples:
      raise ValueError("grow_layer_tuples argument can't be empty.")
    self.grow_layer_tuples = grow_layer_tuples

    def get_n_neuron(n_neuron_initial):#selon leur explication la methode est utilisee pour calculer le nb de neurone a ajouter (n_grow) per layer
      if self._n_grow_fraction:
        
        return int(max(1, n_neuron_initial * self._n_grow_fraction))
      else:
        return self._n_grow
    # Used to calculate n_grow per layer using grow_fraction.
    # n_neurons are decided using the initial architecture.
    self._n_grow_dict = {
        tpl[0].name: get_n_neuron(tpl[0].weights[0].shape[-1])
        for tpl in grow_layer_tuples
    }

  def is_update_iteration(self, iteration): #je pense c'est pour definir si l'iteration courante(arg iteration) est une iteration ou on fait un growth ou pas
    #dans les conditions on voit qu'il faut que l'itertion courante (arg iteration) soit divisible par update_frequency (qui est le nb d'iter entre deux growth)
    assert iteration >= 0
    return ((self.network_grower is not None) and
            (iteration % self._update_frequency == 0) and
            (self._start_iteration <= iteration) and
            ((self._n_growth_steps is None) or
             (self._growth_counter < self._n_growth_steps))) #le nb de growth deja fait (_growth_counter) doit etre inferieur au nb de growth total 
                                                             #a faire (_n_growth_steps) pour faire un nouveau growth sinon on a deja fait tous les growth prevus

  def get_variable_list(self, grow_layer_tuple): #je pense que ca genere un iterateur sur les poids des layers concernes par le growth
    return list(itertools.chain.from_iterable(
        [layer.trainable_weights for layer in grow_layer_tuple]))

  def get_grow_layer_stats(self): #je pense pour recuperer des stats sur les layers concernes par le growth (candidats growth): on retourne le name et le nb de neurone des layer
    all_stats = []
    for grow_layer_tuple in self.grow_layer_tuples:
      first_layer = grow_layer_tuple[0]
      n_neuron = first_layer.get_weights()[0].shape[-1]
      all_stats.append((first_layer.layer.name, n_neuron))
    return all_stats

  def update_network(self, batch_data, optimizer=None):#car cette classe est global elle aura des sous-classes qui vont redefinir cette methode
    raise NotImplementedError()


class DummyUpdater(Updater): # sous-classe de updater (classe precedente, classe globale)
  """Implements common methods.

  Attr:
    network_grower: growers.LayerGrower
    grow_layer_tuples: list of lists<glayers.GrowLayer>, candidates to be
      grown together with their outgoing weights.
    update_frequency: int, Number of iterations before neurons are added.
  """
  #pour les attributs voir les explications de la classe updater c'est pareil
  #je pense cette classe implemente le cas ou on fait pas de growing c'est pour ca qu'on a:
  #==> un pass dans update_network
  #==> un del epoch dans is_update_network parce qu'on supprime les epoch/iter supplementaires qui sont les etapes de growing
  #==> on retourne une liste vide dans get_grow_layer_stats
  def __init__(self, grow_layer_tuples):
    super().__init__(None, grow_layer_tuples, None, None)

  def update_network(self, **kwargs):
    pass

  def is_update_iteration(self, epoch):
    del epoch
    return False

  def get_grow_layer_stats(self):
    return []


class RoundRobin(Updater):
  """Updates provided candidate layers in a round robin fashion."""
  #donc si je pense leur logique c'est qu'on a dans le tuples grow_layer_tuples les layers candidats donc ceux qui peuvent subir le growth
  #avec cette classe on applique le growing en round robin sur ces layers (
  #on leur ajoute pas a tous des neurones mais on choisit le layer a qui on ajoute des neurones selon le round robin

  def _next_grow_layer_tuple(self, unused_batch_data): #retourne pour quel layer faire le growth parmi les candidates layers
    next_tuple_id = self._growth_counter % len(self.grow_layer_tuples)
    self._growth_counter += 1 #car ca compte le nb de growth donc on l'incremente a chaque growth
    return self.grow_layer_tuples[next_tuple_id]

  def update_network(self, batch_data, optimizer=None):
    """Updates the network and optimizer slots."""
    grow_layer_tuple = self._next_grow_layer_tuple(batch_data)
    old_variables = self.get_variable_list(grow_layer_tuple)#methode qui nous retourne un iterateur sur les poids des layer qui subissent le growth
                                                            #donc c'est pour avoir les poids avant de growth
    n_new = self._n_grow_dict[grow_layer_tuple[0].name]
    #la ligne qui suit fait le growing
    self.network_grower.grow_neurons(grow_layer_tuple, batch_data,
                                     n_new=n_new, scale=self._scale) #le network grower c'est defini dans le fichier growers.py et 
                                                                     #c'est ce qui implemente la methode de growing
                                                                     #ca fait appel au fichier layers.py qui cree les layers resultant du growth
    # Run the loss function to create new variables.
    self.compile_fn() #le model.compile de tensorflow
    new_variables = self.get_variable_list(grow_layer_tuple)#methode qui nous retourne un iterateur sur les poids des layer qui subissent le growth
                                                            #donc la c'est pour avoir la liste des poids apres le growth
    optimizer._create_slots(new_variables) #au debut de ce fichier on a definit une fonction qui augmente le shape d'un tensor donne et remboure avec des zero
                                           #plus tard on l'a utilise avec l'optimizer parce qu'on doit faire de la place pour les neurones ajoutes
                                           #donc je pense que cette ligne et les suivantes sont lies a ca
                                           #le but je pense c'est de continuer l'optimisation du loss avec sgd/adam (l'optimizer) apres le growth et il faut
                                           #certain ajustement pour ca et je pense que ces lignes font ca
    if self._carry_optimizer and optimizer:
      self.copy_optimizer_slots(optimizer, old_variables, new_variables)
    self.delete_optimizer_slots(optimizer, old_variables)


class AllAtOnce(Updater):
  """Grows all candidate layers at once."""
  #donc si je pense leur logique c'est qu'on a dans le tuples grow_layer_tuples les layers candidats donc ceux qui peuvent subir le growth
  #avec cette classe on applique le growing sur tous les layers candidats
  def _get_all_grow_layer_tuples(self):
    self._growth_counter += 1 #on incremente car on fait une etape de growing de plus
    return self.grow_layer_tuples[:] #ca veut dire qu'on retourne tous les elements de grow_layer_tuples (donc tous les candidats au growing)

  def update_network(self, batch_data, optimizer=None):
    """Updates the network and optimizer slots."""
    grow_layer_tuples = self._get_all_grow_layer_tuples() #methode precedente on recupere tous les layers candidats au growing
    for grow_layer_tuple in grow_layer_tuples: #on parcout les candidats au growing
      old_variables = self.get_variable_list(grow_layer_tuple)#methode qui nous retourne un iterateur sur les poids des layer qui subissent le growth
                                                              #donc c'est pour avoir les poids avant de growth
      n_new = self._n_grow_dict[grow_layer_tuple[0].name]
      #la ligne qui suit fait le growing
      self.network_grower.grow_neurons(grow_layer_tuple, batch_data,
                                       n_new=n_new, scale=self._scale)#le network grower c'est defini dans le fichier growers.py et 
                                                                     #c'est ce qui implemente la methode de growing
                                                                     #ca fait appel au fichier layers.py qui cree les layers resultant du growth
      # Run the loss function to create new variables.
      self.compile_fn() #le model.compile de tensor flow
      new_variables = self.get_variable_list(grow_layer_tuple) #methode qui nous retourne un iterateur sur les poids des layer qui subissent le growth
                                                              #donc c'est pour avoir les poids apres de growth
      optimizer._create_slots(new_variables) #au debut de ce fichier on a definit une fonction qui augmente le shape d'un tensor donne et remboure avec des zero
                                           #plus tard on l'a utilise avec l'optimizer parce qu'on doit faire de la place pour les neurones ajoutes
                                           #donc je pense que cette ligne et les suivantes sont lies a ca
                                           #le but je pense c'est de continuer l'optimisation du loss avec sgd/adam (l'optimizer) apres le growth et il faut
                                           #certain ajustement pour ca et je pense que ces lignes font ca
      if self._carry_optimizer and optimizer:
        self.copy_optimizer_slots(optimizer, old_variables, new_variables)
      self.delete_optimizer_slots(optimizer, old_variables)




def adjust_epochs(train_epochs, width_scale, update_frequency,
                  start_iteration, n_growth_steps, steps_per_epoch):
  """Adjust the epochs such as the total FLOPs are same as big-baseline."""
  # Here we extend training according to the FLOP saved by starting with
  # a smaller width.
  saved_fraction = (1 - width_scale)
  # Saved before growth.
  saved_steps = saved_fraction * start_iteration
  growth_duration = (update_frequency * (n_growth_steps - 1))
  # Saved during growth (2 is because of the trianble area).
  saved_steps += saved_fraction/2 * growth_duration
  new_epochs = train_epochs + int(saved_steps / steps_per_epoch)
  return new_epochs
