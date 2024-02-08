import tensorflow as tf
from keras.optimizers.schedules import LearningRateSchedule


class BurnInLearningRate(LearningRateSchedule):
        """I wanted to start with a very high lr and then drop it down to a lower one"""
        # might be good to convert this to full class and TF like the lr_schedule
        def __init__(self, initial_learning_rate):
            self.burnin_learning_rate = 0.01 # specify initial learning rate
            self.burnin_epochs = 1  # specify the number of epochs for the burnin

            # these should follow the ExponentialDecay function
            self.learning_rate = initial_learning_rate # specify secondary learning rate
            self.decay_steps = 5  # specify the number of epochs for decaying
            self.decay_rate = 0.96  # specify the decay rate
            self.staircase=True

        def __call__(self, step):
            p = (step - self.burnin_epochs) / (self.decay_steps)
            if self.staircase:
                p = tf.floor(p)
            p = tf.cast(p, tf.float32)
            
            return tf.cond(step < self.burnin_epochs,
                lambda: self.burnin_learning_rate,
                lambda: tf.multiply(self.learning_rate, tf.pow(self.decay_rate, p)))
        
        def get_config(self):
            return {
                "burnin_learning_rate": self.burnin_learning_rate,
                "burnin_epochs": self.burnin_epochs,
                "learning_rate": self.learning_rate,
                "decay_steps": self.decay_steps,
                "decay_rate": self.decay_rate,
                "staircase": self.staircase,
            }