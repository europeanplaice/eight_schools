import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import warnings

def sampling_wrapper(model, y):

    def target_log_prob_fn(avg_effect, avg_stddev, school_effects_standard):
      """Unnormalized target density as a function of states."""
      return model.log_prob((
          avg_effect, avg_stddev, school_effects_standard, y))

    num_results = 5000
    num_burnin_steps = 3000

# Improve performance by tracing the sampler using `tf.function`
# and compiling it using XLA.
    @tf.function(autograph=False, jit_compile=True)
    def do_sampling():
      return tfp.mcmc.sample_chain(
          num_results=num_results,
          num_burnin_steps=num_burnin_steps,
          current_state=[
              tf.zeros([], name='init_avg_effect'),
              tf.zeros([], name='init_avg_stddev'),
              tf.ones([8], name='init_school_effects_standard'),
          ],
          kernel=tfp.mcmc.HamiltonianMonteCarlo(
              target_log_prob_fn=target_log_prob_fn,
              step_size=0.4,
              num_leapfrog_steps=3))

    states, kernel_results = do_sampling()

    avg_effect, avg_stddev, school_effects_standard = states

    school_effects_samples = (
        avg_effect[:, np.newaxis] +
        np.exp(avg_stddev)[:, np.newaxis] * school_effects_standard)

    num_accepted = np.sum(kernel_results.is_accepted)
    print('Acceptance rate: {}'.format(num_accepted / num_results))

    return school_effects_samples
    

def main():
    num_schools = 8
    treatment_effects = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float32)
    treatment_stddevs = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32)

    model = tfd.JointDistributionSequential([
      tfd.Normal(loc=0., scale=10., name="avg_effect"),  # `mu` above
      tfd.Normal(loc=5., scale=1., name="avg_stddev"),  # `log(tau)` above
      tfd.Independent(tfd.Normal(loc=tf.zeros(num_schools),
                                 scale=tf.ones(num_schools),
                                 name="school_effects_standard"),  # `theta_prime` 
                      reinterpreted_batch_ndims=1),
      lambda school_effects_standard, avg_stddev, avg_effect: (
          tfd.Independent(tfd.Normal(loc=(avg_effect[..., tf.newaxis] +
                                          tf.exp(avg_stddev[..., tf.newaxis]) *
                                          school_effects_standard),  # `theta` above
                                     scale=treatment_stddevs),
                          name="treatment_effects",  # `y` above
                          reinterpreted_batch_ndims=1))
    ])

    sampling_wrapper(model, treatment_effects)

main()
