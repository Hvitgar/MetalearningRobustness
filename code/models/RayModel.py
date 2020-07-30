from pba.train import RayModel as RayModelPBA
from pba.model import ModelTrainer as ModelTrainerPBA, Model as ModelPBA
from models.tf.resnet import Resnet18
import pba.helper_utils as helper_utils
import datasets.data_utils as data_utils
import numpy as np
import imagecorruptions as ic

import tqdm

import tensorflow as tf

from datasets import data_utils

class Model(ModelPBA):
    def _build_graph(self, images, labels, mode):
        if self.hparams.model_name in ['resnet18']:
            if self.hparams.model_name == 'resnet18':
                model = Resnet18(self.num_classes)
            is_training = 'train' in mode
            if is_training:
                self.global_step = tf.train.get_or_create_global_step()

            logits = model(images, is_training)
            self.predictions, self.cost = helper_utils.setup_loss(logits, labels)

            self._calc_num_trainable_params()

            # Adds L2 weight decay to the cost
            self.cost = helper_utils.decay_weights(self.cost,
                                                   self.hparams.weight_decay_rate)

            if is_training:
                self._build_train_op()

            # Setup checkpointing for this child model
            # Keep 2 or more checkpoints around during training.
            with tf.device('/cpu:0'):
                self.saver = tf.train.Saver(max_to_keep=10)

            self.init = tf.group(tf.global_variables_initializer(),
                                 tf.local_variables_initializer())
        else:
            super(Model, self)._build_graph(images, labels, mode)

class ModelTrainer(ModelTrainerPBA):
    def __init__(self, hparams):
        self._session = None
        self.hparams = hparams

        # Set the random seed to be sure the same validation set
        # is used for each model
        np.random.seed(0)
        self.data_loader = data_utils.DataSet(hparams)
        np.random.seed()  # Put the random seed back to random
        self.data_loader.reset()
        
        self.mPC = 0

        # extra stuff for ray
        self._build_models()
        self._new_session()
        self._session.__enter__()
    
    def _build_models(self):
        """Builds the image models for train and eval."""
        # Determine if we should build the train and eval model. When using
        # distributed training we only want to build one or the other and not both.
        with tf.variable_scope('model', use_resource=False):
            m = Model(self.hparams, self.data_loader.num_classes, self.data_loader.image_size)
            m.build('train')
            self._num_trainable_params = m.num_trainable_params
            self._saver = m.saver
        with tf.variable_scope('model', reuse=True, use_resource=False):
            meval = Model(self.hparams, self.data_loader.num_classes, self.data_loader.image_size)
            meval.build('eval')
        self.m = m
        self.meval = meval
        
    def _eval_robustness(self, iteration, corruptions=None, severities=None):
        if ((iteration-1) % self.hparams.perturbation_interval == 0):
            """Evaluates corruption robustness of the model"""
            if corruptions is None:
                corruptions = ic.get_corruption_names('validation')
            if severities is None:
                severities = [1,2,3,4,5]
            self.mPC = self.eval_child_model(self.meval, self.data_loader, 'val', robustness=True, corruptions=corruptions, severities=severities)
        tf.logging.info('Validation Robustness: {}'.format(self.mPC))
        return self.mPC

        
    def eval_child_model(self, model, data_loader, mode, robustness=False, corruptions=None, severities=None):
        """Evaluate the child model.

        Args:
          model: image model that will be evaluated.
          data_loader: dataset object to extract eval data from.
          mode: will the model be evalled on train, val or test.

        Returns:
          Accuracy of the model on the specified dataset.
        """
        tf.logging.info('Evaluating child model in mode {}'.format(mode))
        while True:
            try:
                if mode == 'val':
                    loader = self.data_loader.dataloader_val
                elif mode == 'test':
                    loader = self.data_loader.dataloader_test
                else:
                    raise ValueError('Not valid eval mode')
                tf.logging.info('model.batch_size is {}'.format(model.batch_size))
                if robustness:
                    if corruptions is None:
                        corruptions = ic.get_corruption_names()
                    if severities is None:
                        severities = [0,1,2,3,4,5]
                    if mode == 'val':
                        # if mode is 'val', apply a random corruption on a random severity to each image
                        correct = 0
                        count = 0
                        for images, labels in loader:
                            images = np.transpose(images.numpy(), [0,2,3,1])
                            labels = labels.numpy()
                            # produce one-hot target vector
                            labels = np.eye(model.num_classes)[labels]
                            # inverse normalization
                            means = data_loader.augmentation_transforms.MEANS[data_loader.hparams.dataset]
                            stds = data_loader.augmentation_transforms.STDS[data_loader.hparams.dataset]
                            images = ((images * stds) + means) * 255
                            # corrupt
                            images =  images.astype(np.uint8)
                            for j in range(len(images)):
                                s = np.random.choice(severities, 1)[0]
                                if s == 0:
                                    continue
                                c = np.random.choice(corruptions, 1)[0]
                                images[j] = ic.corrupt(images[j], corruption_name=c, severity=s)
                            # normalize
                            images = ((images - means) / stds) / 255.
                            preds = self.session.run(
                                model.predictions,
                                feed_dict={
                                    model.images: images,
                                    model.labels: labels,
                                })
                            correct += np.sum(
                                np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))
                            count += len(preds)
                        assert count == len(loader.dataset)
                        tf.logging.info('correct: {}, total: {}'.format(correct, count))
                        return correct / count
                    else:
                        # if mode is 'test', test all corruptions and severities on each image
                        accuracies = {c: {s: 0 for s in range(6)} for c in corruptions}
                        for c in corruptions:
                            for s in severities:
                                if (s == 0):
                                    if c == corruptions[0]:
                                        # iterate once over the clean dataset
                                        correct = 0
                                        count = 0
                                        progress_bar = tqdm.tqdm(loader)
                                        progress_bar.set_description('Clean')
                                        for images, labels in progress_bar:
                                            images = np.transpose(images.numpy(), [0,2,3,1])
                                            labels = labels.numpy()
                                            # produce one-hot target vector
                                            labels = np.eye(model.num_classes)[labels]
                                            preds = self.session.run(
                                                model.predictions,
                                                feed_dict={
                                                    model.images: images,
                                                    model.labels: labels,
                                                })
                                            correct += np.sum(
                                                np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))
                                            count += len(preds)
                                        assert count == len(loader.dataset)
                                        accuracies[c][s] = correct / count
                                    else:
                                        # clean performance has been evaluated before
                                        # and will just be copied here for convenience
                                        accuracies[c][s] = accuracies[corruptions[0]][s]
                                else:
                                    correct = 0
                                    count = 0

                                    progress_bar = tqdm.tqdm(loader)
                                    progress_bar.set_description('Corruption: {}, Severity: {}'.format(c, s))
                                    for images, labels in progress_bar:
                                        images = np.transpose(images.numpy(), [0,2,3,1])
                                        labels = labels.numpy()
                                        # produce one-hot target vector
                                        labels = np.eye(model.num_classes)[labels]
                                        # inverse normalization
                                        means = data_loader.augmentation_transforms.MEANS[data_loader.hparams.dataset]
                                        stds = data_loader.augmentation_transforms.STDS[data_loader.hparams.dataset]
                                        images = ((images * stds) + means) * 255
                                        # corrupt
                                        images =  images.astype(np.uint8)
                                        for j in range(len(images)):
                                            images[j] = ic.corrupt(images[j], corruption_name=c, severity=s)
                                        # normalize
                                        images = ((images - means) / stds) / 255.

                                        preds = self.session.run(
                                            model.predictions,
                                            feed_dict={
                                                model.images: images,
                                                model.labels: labels,
                                            })
                                        correct += np.sum(
                                            np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))
                                        count += len(preds)
                                    assert count == len(loader.dataset)
                                    accuracies[c][s] = correct / count
                    return accuracies

                else:
                    correct = 0
                    count = 0
                    for images, labels in loader:
                        images = np.transpose(images.numpy(), [0,2,3,1])
                        labels = labels.numpy()
                        # produce one-hot target vector
                        labels = np.eye(model.num_classes)[labels]
                        preds = self.session.run(
                            model.predictions,
                            feed_dict={
                                model.images: images,
                                model.labels: labels,
                            })
                        correct += np.sum(
                            np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))
                        count += len(preds)
                    assert count == len(loader.dataset)
                    tf.logging.info('correct: {}, total: {}'.format(correct, count))
                    accuracy = correct / count
                    tf.logging.info(
                        'Eval child model accuracy: {}'.format(accuracy))
                    # If epoch trained without raising the below errors, break
                    # from loop.
                    break
            except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
                tf.logging.info(
                    'Retryable error caught: {}.  Retrying.'.format(e))

        return accuracy
        

class RayModel(RayModelPBA):
    def _setup(self, *args):
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.logging.info("calling setup")
        self.hparams = tf.contrib.training.HParams(**self.config)
        self.trainer = ModelTrainer(self.hparams)
        
    def _train(self):
        res = super(RayModel, self)._train()
        if self.config['state'] == 'search' and self.config['robustness']:
            corruptions = ic.get_corruption_names('validation')
            robustness_metric = self.trainer._eval_robustness(self._iteration)
            res['robustness'] = robustness_metric
            
        return res