#include functions.stan


data {
    int<lower=1> N;                         // Number of individuals
    array[N] int<lower=1> num_tests;                 // Number of serological tests for each individual
    int<lower=1> num_tests_total; // Total number of serological tests across all individuals
    int<lower=1> K;                         // Number of pathogens
    array[num_tests_total] real test_times; // Time of each serological test
    array[num_tests_total,K] int<lower=0, upper=1> serostatus; // Seropositivity for each test and pathogen
    real interval; // For when all tests are same time apart.
    
    real log_baseline_hazard_mean; // Mean for normal prior on log baseline hazards
    real <lower=0> log_baseline_hazard_scale; // Scale for normal prior on log baseline hazards
    real <lower=0> beta_scale; // scale for Laplace prior on log hazard ratios
}

transformed data {
    int num_infection_states = 1; {
        for (k in 1:K) {num_infection_states *= 2;}
    }
    array[K,K] int beta_indices; // Array mapping pathogen,pathogen pairs to their indices in betas
    {int idx = 0;
    for (i in 1:K) {
        for (j in 1:K) {
            if (i == j) {
                beta_indices[i,j] = 0; // Diagonal elements are arbitrary (no interaction)
            } else {
                idx += 1;
                beta_indices[i,j] = idx;
            }
        }
    }
    }
    /* check the infection state to index mapping
    print("Infection state indices:");
    for (i in 1:num_infection_states) {
        array[K] int infection_state = index_to_infection_state(i, K);
        print(infection_state_to_index(infection_state)); // Should print 1,2,3,...,2^K
    }
    */
}
parameters {
    array[K] real<lower=0> baseline_hazards;    // Constant baseline hazard λ₀
    array[K] real<lower=0> seroreversion_rates; // Seroreversion rates for each pathogen
    array[K*(K-1)] real betas;                  // Log hazard ratios for pathogen, pathogen pair where the pathogens are distinct
}

transformed parameters {
    matrix[K,K] beta_matrix; // Hazard matrix for pairwise interactions
    {int idx = 1;
    for (i in 1:K) {
        for (j in 1:K) {
            if (i == j) {
                beta_matrix[i,j] = 0.0; // Diagonal elements must be zero for calculation to work
            } else {
                beta_matrix[i,j] = betas[idx];
                idx += 1;
            }
        }
    }}
    matrix[num_infection_states,num_infection_states] q_matrix = rep_matrix(
        0.0, num_infection_states, num_infection_states
    ); {
        // Fill the q_matrix with transition rates
        for (i in 1:num_infection_states) {
            array[num_infection_states] int infection_state_i = index_to_infection_state(i, num_infection_states);
            real row_sum = 0.0;
            for (k in 1:K) {
                array[num_infection_states] int infection_state_j = infection_state_i;
                infection_state_j[k] = 1 - infection_state_i[k]; // Toggle the k-th pathogen state
                int j = infection_state_to_index(infection_state_j);
                if (infection_state_i[k] == 1) {
                    // If the individual is infected with pathogen k, we can transition to the state where they are not infected with pathogen k
                    q_matrix[i,j] = seroreversion_rates[k]; // Transition rate from i to j
                    row_sum += seroreversion_rates[k];
                } else {
                    // If the individual is not infected with pathogen k, we can transition to the state where they are infected with pathogen k
                    q_matrix[i,j] = baseline_hazards[k]; // Transition rate from i to j
                    for (l in 1:K) {
                        if (infection_state_i[l] == 1) {
                            q_matrix[i,j] *= exp(beta_matrix[l,k]); // Apply the hazard ratio
                        }
                    }
                    row_sum += q_matrix[i,j];
                }
            }
            q_matrix[i,i] = -row_sum; // Diagonal element is the negative sum of the row
        }
    }
    // matrix[num_infection_states,num_infection_states] q_power_2 = q_matrix * q_matrix; // Square the transition rate matrix
    // matrix[num_infection_states,num_infection_states] q_power_3 = q_power_2 * q_matrix; // Cube the transition rate matrix
    // matrix[num_infection_states,num_infection_states] q_power_4 = q_power_3 * q_matrix; // Raise the transition rate matrix to the fourth power
    // matrix[num_infection_states,num_infection_states] eye = diag_matrix(rep_vector(1.0, num_infection_states)); // Identity matrix
    // matrix[num_infection_states,num_infection_states] transition_matrix = matrix_exp(interval * q_matrix);
}

model {
    // Priors
    target += -log(baseline_hazards); // log(1/λ₀) = -log(λ₀)
    // baseline_hazards ~ lognormal(log_baseline_hazard_mean, log_baseline_hazard_scale); // Log-normal prior on baseline hazards
    betas ~ double_exponential(0, beta_scale);
    // target += -log(seroreversion_rates); // TO DO: make this uninformative
    seroreversion_rates ~ normal(0, 1); // Normal prior on seroreversion rates

    // Likelihood
    array[N] real log_lik = rep_array(0.0, N); {
        int test_idx = 1; // Index for the current test
        array[K] int prev_serostatus;
        real prev_test_time;
        int prev_state_index;
        matrix[num_infection_states,1] prev_infection_state_vector;
        array[K] int next_serostatus;
        real next_test_time;
        int next_state_index;
        matrix[num_infection_states,1] next_infection_state_vector;
        for (i in 1:N) {
            prev_serostatus = serostatus[test_idx,]; // Initial serostatus for the individual
            prev_test_time = test_times[test_idx]; // Initial test time for the individual
            prev_state_index = infection_state_to_index(prev_serostatus);
            prev_infection_state_vector = rep_matrix(0.0, num_infection_states, 1);
            prev_infection_state_vector[prev_state_index,1] = 1.0; // Set the initial state vector
            test_idx += 1;
            for (j in 1:num_tests[i]-1) {
                // Get the current test time and serostatus
                next_serostatus = serostatus[test_idx];
                next_test_time = test_times[test_idx];
                next_state_index = infection_state_to_index(next_serostatus);
                next_infection_state_vector = rep_matrix(0.0, num_infection_states, 1);
                next_infection_state_vector[next_state_index,1] = 1.0;
                // log_lik[i] += log(transition_matrix[prev_state_index,next_state_index]);
                log_lik[i] += log(
                    matrix_exp(
                        q_matrix * (next_test_time - prev_test_time) // q_matrix times time difference between tests
                    )[prev_state_index, next_state_index]
                );
                // log_lik[i] += log(transition_prob_approx(
                //     prev_state_index,
                //     next_state_index,
                //     next_test_time - prev_test_time,
                //     q_matrix, q_power_2, q_power_3, q_power_4, eye
                // ));
                // Update the previous state for the next iteration
                prev_serostatus = next_serostatus;
                prev_test_time = next_test_time;
                prev_state_index = next_state_index;
                prev_infection_state_vector = next_infection_state_vector;
                test_idx += 1;
            }
        }
    }
    target += log_lik;
}

generated quantities {
    // print(q_matrix);
    // Likelihood
    array[N] real log_lik = rep_array(0.0, N); {
        int test_idx = 1; // Index for the current test
        array[K] int prev_serostatus;
        real prev_test_time;
        int prev_state_index;
        matrix[num_infection_states,1] prev_infection_state_vector;
        array[K] int next_serostatus;
        real next_test_time;
        int next_state_index;
        matrix[num_infection_states,1] next_infection_state_vector;
        for (i in 1:N) {
            prev_serostatus = serostatus[test_idx,]; // Initial serostatus for the individual
            prev_test_time = test_times[test_idx]; // Initial test time for the individual
            prev_state_index = infection_state_to_index(prev_serostatus);
            prev_infection_state_vector = rep_matrix(0.0, num_infection_states, 1);
            prev_infection_state_vector[prev_state_index,1] = 1.0; // Set the initial state vector
            test_idx += 1;
            for (j in 1:num_tests[i]-1) {
                // Get the current test time and serostatus
                next_serostatus = serostatus[test_idx];
                next_test_time = test_times[test_idx];
                next_state_index = infection_state_to_index(next_serostatus);
                next_infection_state_vector = rep_matrix(0.0, num_infection_states, 1);
                next_infection_state_vector[next_state_index,1] = 1.0;
                // log_lik[i] += log(transition_matrix[prev_state_index,next_state_index]);
                log_lik[i] += log(
                    matrix_exp(
                        q_matrix * (next_test_time - prev_test_time) // q_matrix times time difference between tests
                    )[prev_state_index, next_state_index]
                );
                // log_lik[i] += log(transition_prob_approx(
                //     prev_state_index,
                //     next_state_index,
                //     next_test_time - prev_test_time,
                //     q_matrix, q_power_2, q_power_3, q_power_4, eye
                // ));
                // Update the previous state for the next iteration
                prev_serostatus = next_serostatus;
                prev_test_time = next_test_time;
                prev_state_index = next_state_index;
                prev_infection_state_vector = next_infection_state_vector;
                test_idx += 1;
            }
        }
    }
}