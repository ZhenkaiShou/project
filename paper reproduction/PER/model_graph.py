import tensorflow as tf

from model import QValueNetwork

def huber_loss(input, delta = 1.0):
  return tf.where(tf.abs(input) <= delta, 0.5 * tf.square(input), delta * (tf.abs(input) - 0.5 * delta))

class ModelGraph(object):
  def __init__(self, obs_shape, num_action, network_type, gamma = 1.0, grad_max_norm = 10):
    # Build networks.
    self.Obs = tf.placeholder(tf.float32, (None, *obs_shape))
    self.ObsTarget = tf.placeholder(tf.float32, (None, *obs_shape))
    main_network = QValueNetwork(self.Obs, num_action, network_type, name = "main_network")
    target_network = QValueNetwork(self.ObsTarget, num_action, network_type, name = "target_network")
    
    # Update target network.
    self.update_target_network_op = [tf.assign(ref, value) for ref, value in zip(target_network.variables, main_network.variables)]
    
    # Sample actions.
    self.Epsilon = tf.placeholder(tf.float32, ())
    batch_size = tf.shape(self.Obs)[0]
    random_action = tf.random_uniform(tf.stack([batch_size]), minval = 0, maxval = num_action, dtype=tf.int64)
    self.greedy_action = tf.argmax(main_network.q, axis = 1)
    random_value = tf.random.uniform(tf.stack([batch_size]), minval = 0, maxval = 1, dtype = tf.float32)
    self.epsilon_action = tf.where(random_value < self.Epsilon, random_action, self.greedy_action)
    
    # Train.
    self.Action = tf.placeholder(tf.int32, (None,))
    self.Reward = tf.placeholder(tf.float32, (None,))
    self.Done = tf.placeholder(tf.float32, (None,))
    self.Weights = tf.placeholder(tf.float32, (None,))
    
    # Double Q-learning.
    #   a_next = argmax(q_main_network(obs_next))
    #   q_target = reward + (1 - done) * gamma * q_target_network(obs_next, a_next)
    main_network_for_target_evaluation = QValueNetwork(self.ObsTarget, num_action, network_type, name = "main_network")
    action_next = tf.argmax(main_network_for_target_evaluation.q, axis = 1)
    q_next = tf.reduce_sum(target_network.q * tf.one_hot(action_next, num_action), axis = 1)
    q_target = self.Reward + (1 - self.Done) * gamma * q_next
    
    # Loss function.
    q_a = tf.reduce_sum(main_network.q * tf.one_hot(self.Action, num_action), axis = 1)
    self.td_error = q_target - q_a
    loss = tf.reduce_mean(self.Weights * huber_loss(self.td_error, delta = 1.0))
    
    # Optimization.
    self.LearningRate = tf.placeholder(tf.float32, ())
    optimizer = tf.train.AdamOptimizer(self.LearningRate)
    gradients = optimizer.compute_gradients(loss, var_list = main_network.trainable_variables)
    gradients = [(tf.clip_by_norm(grad, grad_max_norm), var) for grad, var in gradients]
    self.train_op = optimizer.apply_gradients(gradients)
  
  def initialize_variables(self):
    # Initialize variables.
    tf.get_default_session().run(tf.global_variables_initializer())
  
  def update_target_network(self):
    # Update target network.
    tf.get_default_session().run(self.update_target_network_op)
  
  def epsilon_act(self, obs, epsilon):
    # Sample action with epsilon-greedy policy.
    epsilon_action = tf.get_default_session().run(self.epsilon_action, feed_dict = {self.Obs: obs, self.Epsilon: epsilon})
    
    return epsilon_action
  
  def act(self, obs):
    # Act greedily.
    greedy_action = tf.get_default_session().run(self.greedy_action, feed_dict = {self.Obs: obs})
    
    return greedy_action
  
  def train(self, obs, action, reward, done, obs_next, weights, learning_rate):
    # One step optimization.
    _, td_error = tf.get_default_session().run([self.train_op, self.td_error], 
      feed_dict = {
        self.Obs: obs, self.Action: action, self.Reward: reward, self.Done: done, 
        self.ObsTarget: obs_next, self.Weights: weights, self.LearningRate: learning_rate
      })
    
    return td_error
  
  def save(self, path):
    # Save the main network.
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "main_network")
    saver = tf.train.Saver(var_list = var_list)
    saver.save(tf.get_default_session(), path)
  
  def load(self, path):
    # Load the main network.
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "main_network")
    saver = tf.train.Saver(var_list = var_list)
    saver.restore(tf.get_default_session(), path)