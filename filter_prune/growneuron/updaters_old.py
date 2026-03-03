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
import numpy as np


def pad_zeros_to(tensor, new_shape, n_to_remove, list_filters_to_remove):
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
    diff_shape = [old_shape[-1] - new_shape[-1]] #-1 indexe le dernier element du tuple shape
    concat_axis = -1
  else:
    if old_shape[-2] == new_shape[-2]:
      # Input features are same, padding at axis=-1.
      concat_axis = -1
    else:
      concat_axis = -2
    diff_shape = list(old_shape)
    diff_shape[concat_axis] = old_shape[concat_axis] - new_shape[concat_axis]

    split_tensor=np.split(tensor, tensor.shape[concat_axis], axis=len(tensor.shape)+concat_axis)
    split_tensor=np.array(split_tensor)
    

    split_tensor=np.delete(split_tensor, list_filters_to_remove, axis=0)
    split_tensor=np.array(split_tensor)
    split_tensor=split_tensor.tolist()
    
  return np.concatenate(split_tensor, axis=len(tensor.shape)+concat_axis)


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
    
    
    carry_optimizer: bool, If true the running averages are carried to the new
      optimizer after the growth. Since variables are recreated after growth
      this is necessary. #je pense qu'ils font ca parce que a chaque grow le layer qui subit le growth est remplace par un nouveau donc je pense qu'ils passent le
                         #running average a l'optimizer pour ne pas perturber le gradient descent
  """

  def __init__(self, network_grower, grow_layer_tuples, loss_fn=lambda x: x,
               compile_fn=lambda: None, n_grow=1,
               n_grow_fraction=None, n_growth_steps=None,
               carry_optimizer=True): #pour la signification des arguments voir leur comment c'est clair
    
    self._carry_optimizer = carry_optimizer
    self._n_grow = n_grow
    self._n_grow_fraction = n_grow_fraction

    
    
    
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

  def copy_optimizer_slots(self, optimizer, old_variables, new_variables, n_to_remove, list_filters_to_remove):
    """Copy old slots and pad with zeros for new neurons."""
    #on utilise la fonction pad_zeros_to definit en haut du code pour rembourer avec des zero, je pense les zero ajoutes font de la place pour les nouveau neurones
    for old_var, new_var in zip(old_variables, new_variables):
      for s_name in sorted(optimizer.get_slot_names()):
        old_slot_var = optimizer.get_slot(old_var, s_name)
        new_slot_var = optimizer.get_slot(new_var, s_name)
        # This is used to retrieve the part of the new slot used for the
        # old variables. This assumes new variables are appended to the end.
        new_slot_values = pad_zeros_to(old_slot_var, new_slot_var.shape, n_to_remove, list_filters_to_remove)
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
        return self.n_grow
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
      n_to_remove = self._n_grow_dict[grow_layer_tuple[0].name]
      n_to_remove=int(n_to_remove * self._n_grow_fraction) #on doit multiplier par n_grow_fraction pour avoir le nb de filtres à élaguer égal au nb de filtres ajoutés dans gradmax
                                                     #on doit faire ça car dans gradmax on démarre d'une archi plus petite donc avec un nb de filtres plus petit
                                                     
      print(n_to_remove)
      
      list_filters_to_remove=self.network_grower.prune_filters_l1_norm(grow_layer_tuple, n_to_remove=n_to_remove)
      #list_filters_to_remove=self.network_grower.prune_filters_random(grow_layer_tuple, n_to_remove=n_to_remove)
      #la ligne qui suit fait le growing
      self.network_grower.grow_neurons(grow_layer_tuple, batch_data,
                                       n_to_remove=n_to_remove, list_filters_to_remove=list_filters_to_remove)#le network grower c'est defini dans le fichier growers.py et 
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
        self.copy_optimizer_slots(optimizer, old_variables, new_variables, n_to_remove, list_filters_to_remove)
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
