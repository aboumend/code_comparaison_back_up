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

"""Wide ResNet 28-10 on CIFAR-10/100 trained with maximum likelihood.

Hyperparameters differ slightly from the original paper's code
(https://github.com/szagoruyko/wide-residual-networks) as TensorFlow uses, for
example, l2 instead of weight decay, and a different parameterization for SGD's
momentum.

"""

import itertools
import os
import time

from absl import app
from absl import flags
from absl import logging

import data
#from growneuron.cifar import vgg
import vgg
#from growneuron.cifar import wide_resnet
import wide_resnet

from ml_collections import config_flags #ml_collections est une library ou on a des configdict ou on garde les param des expemiments
#je pense qu'il l'utilise pour les hyperparam
#le config_flags est utilise pour definir la config soit depuis un fichier ou directement
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines.schedules as ub_schedules
from tensorboard.plugins.hparams import api as hp

import random

config_flags.DEFINE_config_file(
    name='config',
    default='growneuron\configs\\baseline_big.py', #par defaut si l'argument --config n'est pas renseigne
    help_string='training config file.')
# common flags

flags.DEFINE_string('data_dir', None,
                    'data_dir to be used for tfds dataset construction.'
                    'It is required when training with cloud TPUs')
flags.DEFINE_bool('download_data', False,
                  'Whether to download data locally when initializing a '
                  'dataset.')
flags.DEFINE_string('output_dir', '/tmp/cifar', 'Output directory.')
flags.DEFINE_bool('collect_profile', False,
                  'Whether to trace a profile with tensorboard')

FLAGS = flags.FLAGS


def get_optimizer(optimizer_config, train_epochs, batch_size, steps_per_epoch): #definit l'optimizer avec le lr schedule si necessaire
  """Given the config and training arguments returns an optimizer."""
  # Linearly scale learning rate and the decay epochs by vanilla settings.
  base_lr = optimizer_config.base_learning_rate * batch_size / 128
  lr_decay_epochs = [int(fraction * train_epochs)
                     for fraction in optimizer_config.lr_decay_epochs]
  if optimizer_config.decay_type == 'step':
    lr_schedule = ub_schedules.WarmUpPiecewiseConstantSchedule(
        steps_per_epoch,
        base_lr,
        decay_ratio=optimizer_config.lr_decay_ratio,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=optimizer_config.lr_warmup_epochs)
  elif optimizer_config.decay_type == 'cosine':
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        base_lr, train_epochs * steps_per_epoch, alpha=0.0)
  else:
    lr_schedule = base_lr / 100.
    logging.info('No decay used')
  optimizer = tf.keras.optimizers.SGD(lr_schedule,
                                      momentum=optimizer_config.momentum,
                                      nesterov=optimizer_config.nesterov)
  return optimizer


def main(argv):
  selectivity=0.5
  

  fmt = '[%(filename)s:%(lineno)s] %(message)s'
  formatter = logging.PythonFormatter(fmt)
  logging.get_absl_handler().setFormatter(formatter)
  del argv  # unused arg
  config = FLAGS.config #pour recuperer la config donc les hyperparam

  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  #tf.random.set_seed(config.seed)
  print("training with data selection, selectivity=", selectivity)

  #strategy = tf.distribute.MirroredStrategy() #normalement c'est pour une execution distribuee
  strategy=tf.distribute.get_strategy() #pour avoir une execution sans distribution on utilise la strategie par defaut qu'on recupere avec 
                                        #tf.distribute.get_strategy, avec ca c'est comme si on utilise aucune strategy, le code lie a ca sera juste ignore
  
  ds_builder = tfds.builder(config.dataset) #cree un builder pour le dataset dont le nom est contenu dans config.dataset
  if FLAGS.download_data:
    ds_builder.download_and_prepare() #pour telecharger le dataset et le mettre sur disque
  ds_info = ds_builder.info #pour documenter le dataset avoir des info dessus (nb exemple, nb classes)
  batch_size = config.per_core_batch_size * config.num_cores #pour savoir combien de batch par coeur (voir fichier config/baseline_small pour voir les valeurs des param)

  # Scale arguments that depend on 128 batch size total training iterations.
  multiplier = 128. / batch_size
  if hasattr(config.updater, 'update_frequency'):
    config.updater.update_frequency = int(
        config.updater.update_frequency * multiplier)
    config.updater.start_iteration = int(
        config.updater.start_iteration * multiplier)

  train_dataset_size = ds_info.splits['train'].num_examples #recuperer le nb de training examples
  steps_per_epoch = train_dataset_size // batch_size
  logging.info('Steps per epoch %s', steps_per_epoch)
  logging.info('Size of the dataset %s', train_dataset_size)
  steps_per_eval = ds_info.splits['test'].num_examples // batch_size
  num_classes = ds_info.features['label'].num_classes

  train_dataset = strategy.distribute_datasets_from_function(#si on fait un training distribue
      data.build_input_fn(ds_builder, batch_size, topology=None, #ATTENTION: a garder meme si on fait pas du distribue, c'est la partie pretraitement du dataset
                          is_training=True,
                          cache_dataset=config.cache_dataset))
  
  test_dataset = strategy.distribute_datasets_from_function( #meme chose mais pour le test
      data.build_input_fn(ds_builder, batch_size, topology=None,#ATTENTION: a garder meme si on fait pas du distribue, c'est la partie pretraitement du dataset
                          is_training=False,
                          cache_dataset=config.cache_dataset))
  # Scale the trianing epochs to match roughly to big-baseline cost.
  arch_name = config.get('architecture', 'wide-resnet') #choix archi, par defaut c'est wide-resnet
  
  grower=None
  if config.scale_epochs:# je pense: Scale the trianing epochs to match roughly to big-baseline cost.
    if arch_name == 'wide-resnet':
      width_scale = config.model.block_width_multiplier
    elif arch_name == 'vgg':
      width_scale = config.model.width_multiplier
    else:
      raise ValueError(f'Unknown architecture: {arch_name}')
    old_epochs = config.train_epochs
    ############# IMPORTANT: si o n'a pas de gain en energie essayer d'enlever l'ajustement des epochs
    # Adjust the total epochs to match big-baseline training FLOPs.
    if grower:
      config.train_epochs = updaters.adjust_epochs( #voir la fonction du fichier updaters.py, le but est d'avoir le meme nb de flop que baseline big
          config.train_epochs,
          width_scale,
          config.updater.update_frequency,
          config.updater.start_iteration,
          config.updater.n_growth_steps,
          steps_per_epoch
          ) #ils rajoutent des flops pour avoir le meme nb que baselinebig car le fait de commencer avec de petit modele reduit les flops (voir la fonction adjust_epochs)
          #dans le fichier updaters.py
    else:
      # baseline
      config.train_epochs = config.train_epochs / width_scale
    logging.info('Extended training from %s to %s', old_epochs,
                 config.train_epochs)
  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with summary_writer.as_default():
    flat_param_dict = {}
    def flat_fn(dic):
      for k, v in dic.items():
        if isinstance(v, dict):
          flat_fn({f'{k}.{k2}': v2 for k2, v2 in v.items()})
        else:
          flat_param_dict[k] = str(v) if isinstance(v, list) else v
    flat_fn(config.to_dict())
    hp.hparams(flat_param_dict)

  
  with strategy.scope():
    if arch_name == 'wide-resnet':
      logging.info('Building ResNet model')
      model = wide_resnet.create_model( #voir fichier cifar/wide_resnet.py c'est la definition du modele
          num_classes=num_classes,
          seed=config.seed,
          **config.model)
      
    elif arch_name == 'vgg': #voir fichier vgg, c'est la definition du modele
      logging.info('Building VGG model')
      model = vgg.create_model(
          num_classes=num_classes,
          seed=config.seed,
          **config.model)
      
    else:
      raise ValueError(f'Unknown architecture: {arch_name}')
    
    # Initialize the parameters.
    def compile_model_fn(): #c'est pour l'init des param, un peu comme le model.build, car quand on cree un modele les param sont vides
      model(tf.keras.Input((32, 32, 3)))
    compile_model_fn()
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params()) #ca permet de voir la dif avant et apres growing
    
    optimizer = get_optimizer(config.optimizer, config.train_epochs, batch_size,
                              steps_per_epoch) #on appelle get_optimizer pour definir l'optimizer
    train_metrics = {
        'train/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'train/accuracy':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss':
            tf.keras.metrics.Mean(),
    }

    eval_metrics = {
        'test/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'test/accuracy':
            tf.keras.metrics.SparseCategoricalAccuracy(),
    }
    model.summary()

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint: #pour redemarrer depuis un checkpoint attention a leur to do je pense il manque des choses
      # TODO This probably wouldn't work if the networks is grown;
      # so we need to switch to saved models maybe.
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

    def loss_fn(inputs): #calcul loss
      images, labels = inputs
      logits = model(images, training=True)
      one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), num_classes)
      loss = tf.reduce_mean(
          tf.keras.losses.categorical_crossentropy(
              one_hot_labels,
              logits,
              from_logits=True))
      scaled_loss = loss / strategy.num_replicas_in_sync #pour le training distribue a enlever je pense
      # Don't add the regularization as unnecessary for zero variables.
      return scaled_loss

    

  

  def train_step(iterator, flag_select):
    """Training StepFn."""
    # This allows retracing. We need retrace as model is changing.
   
    ss=0
    nb_selected_batches_in=0
    nb_not_selected_batches_in=0
    nb_total_iters_in=0
    for ind in tf.range(tf.cast(steps_per_epoch, tf.int32)):#parcours des batches/iteration de l'epoch
      nb_total_iters_in+=1
      print("step", ss)
      
      rand_flag = random.uniform(0, 1) > selectivity

      if (not(flag_select)):
        if (ind==0):
          flag_select=False #si c'est l'iter 0 de l'epoch 0 pas de selection
          print("first iter of training, setting selection flag to false")
        else:
          flag_select=True

      print("iteration", ind, "flag_select", flag_select)

      if rand_flag:
        if (flag_select):
            nb_not_selected_batches_in+=1
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            continue

      nb_selected_batches_in+=1
      images, labels = inputs
      with tf.GradientTape() as tape:
        nb_selected_batches+=1
        logits = model(images, training=True)
        one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), num_classes)
        nll_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                one_hot_labels,
                logits,
                from_logits=True))
        l2_loss = sum(model.losses)
        loss = nll_loss + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync
      grads = tape.gradient(scaled_loss, model.trainable_variables)
      # Logging some gradient norms
      
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      train_metrics['train/loss'].update_state(loss)
      train_metrics['train/negative_log_likelihood'].update_state(nll_loss)
      train_metrics['train/accuracy'].update_state(labels, logits)
      
      # Logging
      if optimizer.iterations % config.get('log_freq', 100) == 1:
        logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                     train_metrics['train/loss'].result(),
                     train_metrics['train/accuracy'].result() * 100)
        total_results = {name: metric.result() for name, metric in
                         train_metrics.items()}
        total_results['lr'] = optimizer.learning_rate(optimizer.iterations)
        total_results['params/total'] = model.count_params()
        
        with summary_writer.as_default():
          for name, result in total_results.items():
            tf.summary.scalar(name, result, step=optimizer.iterations)
        for metric in itertools.chain(train_metrics.values()):
          metric.reset_states()

    return nb_total_iters_in, nb_selected_batches_in, nb_not_selected_batches_in

  def test_step(iterator, dataset_split, num_steps):
    """Evaluation StepFn."""
    @tf.function
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      logits = model(images, training=False)
      probs = tf.nn.softmax(logits)
      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

      eval_metrics[f'{dataset_split}/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      eval_metrics[f'{dataset_split}/accuracy'].update_state(labels, probs)

    for _ in tf.range(tf.cast(num_steps, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  train_iterator = iter(train_dataset)
 

  start_time = time.time()
  tb_callback = None
  if FLAGS.collect_profile:
    tb_callback = tf.keras.callbacks.TensorBoard(
        profile_batch=(100, 2000),
        log_dir=os.path.join(FLAGS.output_dir, 'logs'))
    tb_callback.set_model(model)

  
  nb_total_iters=0
  nb_selected_batches=0
  nb_not_selected_batches=0
  flag_select=False #utilise pour la premiere iter de la premiere epoch, donc le premier batch du training, ce batch doit etre selectionne etant donne qu'on fait
                    #la step de l'optimizer sur les gradients du batch precedent quand le batch actuel n'est pas selectionne
                    #donc le premier batch doit toujours etre pris, si on ne le fait on aura une erreur car il n'y aura pas de gradients de batch precedent sur lesquels
                    #faire la step de l'optimizer
  for epoch in range(initial_epoch, config.train_epochs): #parcours des epochs, boucle de training globale
    logging.info('Starting to run epoch: %s', epoch)
    
    if tb_callback:
      tb_callback.on_epoch_begin(epoch)
    train_start_time = time.time()

    if (epoch==initial_epoch):
      flag_select=False
    else:
      flag_select=True



    #on appelle la fonction train_step qui parcours les batch(donc boucle interne a l'epoch)
    nb_iters, nb_selected, nb_not_selected= train_step(train_iterator, flag_select) #le train_iterator iter sur le train_set


    nb_total_iters+=nb_iters
    nb_selected_batches+=nb_selected
    nb_not_selected_batches+=nb_not_selected


    train_ms_per_example = (time.time() - train_start_time) * 1e6 / batch_size
    print("train step done")
    current_step = (epoch + 1) * steps_per_epoch
    max_steps = steps_per_epoch * config.train_epochs
    time_elapsed = time.time() - start_time
    steps_per_sec = float(current_step) / time_elapsed
    eta_seconds = (max_steps - current_step) / steps_per_sec
    message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
               'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                   current_step / max_steps,
                   epoch + 1,
                   config.train_epochs,
                   steps_per_sec,
                   eta_seconds / 60,
                   time_elapsed / 60))
    logging.info(message)
    if tb_callback:
      tb_callback.on_epoch_end(epoch)

    test_iterator = iter(test_dataset)
    logging.info('Starting to run eval at epoch: %s', epoch)
    test_start_time = time.time()
    test_step(test_iterator, 'test', steps_per_eval)
    test_ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
    logging.info('Done with eval on')

    logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                 eval_metrics['test/negative_log_likelihood'].result(),
                 eval_metrics['test/accuracy'].result() * 100)
    total_results = {name: metric.result() for name, metric
                     in eval_metrics.items()}
    total_results['train/ms_per_example'] = train_ms_per_example
    total_results['test/ms_per_example'] = test_ms_per_example
    with summary_writer.as_default():
      for name, result in total_results.items():
        tf.summary.scalar(name, result, step=epoch + 1)

    for metric in eval_metrics.values():
      metric.reset_states()

    if (config.checkpoint_interval > 0 and
        (epoch + 1) % config.checkpoint_interval == 0):
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)

  final_checkpoint_name = checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
  logging.info('Saved last checkpoint to %s', final_checkpoint_name)

  print("nb total batches", nb_total_iters, "nb selected batches", nb_selected_batches, "nb not selected batches", nb_not_selected_batches) 

if __name__ == '__main__':
  app.run(main)
