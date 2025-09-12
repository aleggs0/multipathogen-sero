functions{
    int infection_state_to_index(
        array[] int infection_state
    ) {
        // Convert an infection state (array of 0s and 1s) to an index in the range [1, num_infection_states]
        int state_index = 1; // 1-indexed
        int pow_2 = 1;
        for (k in 1:num_elements(infection_state)) {
            state_index += infection_state[k] * pow_2;
            pow_2 *= 2; // Each infection state is a binary digit, so we multiply by 2 for the next position
        }
        return state_index;
    }

    array[] int index_to_infection_state(
        int state_index,
        int K
    ) {
        // Convert an state_index in the range [1, num_infection_states] to an infection state (array of 0s and 1s)
        int remainder = state_index - 1; // Convert to 0-indexed
        array[K] int infection_state;
        for (k in 1:K) {
            if (remainder % 2 == 1) {
                infection_state[k] = 1; // Set the k-th element to 1 if the k-th bit is set
            } else {
                infection_state[k] = 0; // Set the k-th element to 0 if the k-th bit is not set
            }
            remainder = remainder %/% 2; // Shift right by one bit
        }
        return infection_state;
    }

    array[] int boolean_index(array[] int data_array, array[] int boolean_array) {
        int len_result = sum(boolean_array);
        array[len_result] int result;
        int counter = 0;
        for (k in 1:num_elements(data_array)) {
            if (boolean_array[k] == 1) {
                counter += 1;
                result[counter] = data_array[k];
            }
        }
        return result;
    }

    real transition_prob_approx(
        int from_state_index,
        int to_state_index,
        real time_interval,
        matrix q_matrix,
        matrix q_power_2,
        matrix q_power_3,
        matrix q_power_4,
        matrix eye
    )
    {
        // Approximate the transition probability from one state to another over a time interval
        // using a [truncated] Taylor series expansion for the matrix exponential.
        // This is a simplified version that uses powers of the transition rate matrix.
        return
            eye[from_state_index, to_state_index] +
            time_interval * q_matrix[from_state_index, to_state_index] +
            (time_interval^2 / 2) * q_power_2[from_state_index, to_state_index] +
            (time_interval^3 / 6) * q_power_3[from_state_index, to_state_index] +
            (time_interval^4 / 24) * q_power_4[from_state_index, to_state_index];
    }
}


data {
    int<lower=2, upper=2> K;                         // Number of pathogens
    
    int<lower=1> N;                         // Number of individuals
    array[N] int<lower=1> num_obs;                 // Number of serological tests for each individual
    int<lower=1> num_obs_total; // Total number of serological tests across all individuals
    array[num_obs_total] real obs_times; // Time of each serological test
    array[num_obs_total,K] int<lower=0, upper=1> serostatus; // Seropositivity for each test and pathogen
    
    int<lower=1> N_test;                         // Number of individuals
    array[N_test] int<lower=1> num_obs_test;                 // Number of serological tests for each individual
    int<lower=1> num_obs_total_test; // Total number of serological tests across all individuals
    array[num_obs_total_test] real obs_times_test; // Time of each serological test
    array[num_obs_total_test,K] int<lower=0, upper=1> serostatus_test; // Seropositivity for each test and pathogen
    int<lower=1> n_frailty_samples; // Number of frailty samples to draw in generated quantities

    real log_baseline_hazard_mean; // Mean for normal prior on log baseline hazards
    real <lower=0> log_baseline_hazard_scale; // Scale for normal prior on log baseline hazards
    real <lower=0> beta_scale; // scale for Laplace prior on log hazard ratios
    real <lower=0> seroreversion_rate_scale; // scale for half-normal prior on seroreversion rates
    real <lower=0> log_frailty_std_scale; // scale for half-normal prior on scale of individual frailties, sigma_u
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
    array[K] real<lower=0> baseline_hazards;     // Constant baseline hazard λ₀
    array[K] real<lower=0> seroreversion_rates; // Seroreversion rates for each pathogen
    array[K*(K-1)] real betas;                  // Log hazard ratios for pathogen, pathogen pair where the pathogens are distinct
    real<lower=0> log_frailty_std;             // Scale of individual frailties
    array[N] real<lower=0> log_frailty_deviations;           // Individual frailties
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
    matrix[num_infection_states,num_infection_states] seroreversion_rate_matrix = rep_matrix(
        0.0, num_infection_states, num_infection_states
    ); {
        seroreversion_rate_matrix[infection_state_to_index({1,0}),infection_state_to_index({0,0})] = seroreversion_rates[1];
        seroreversion_rate_matrix[infection_state_to_index({1,1}),infection_state_to_index({0,1})] = seroreversion_rates[1];
        seroreversion_rate_matrix[infection_state_to_index({0,1}),infection_state_to_index({0,0})] = seroreversion_rates[2];
        seroreversion_rate_matrix[infection_state_to_index({1,1}),infection_state_to_index({1,0})] = seroreversion_rates[2];
        seroreversion_rate_matrix[infection_state_to_index({1,0}),infection_state_to_index({1,0})] = -seroreversion_rates[1];
        seroreversion_rate_matrix[infection_state_to_index({0,1}),infection_state_to_index({0,1})] = -seroreversion_rates[2];
        seroreversion_rate_matrix[infection_state_to_index({1,1}),infection_state_to_index({1,1})] = -seroreversion_rates[1] - seroreversion_rates[2];
    }

    matrix[num_infection_states,num_infection_states] baseline_seroconversion_rate_matrix = rep_matrix(
        0.0, num_infection_states, num_infection_states
    ); {
        baseline_seroconversion_rate_matrix[infection_state_to_index({0,0}),infection_state_to_index({1,0})] = baseline_hazards[1];
        baseline_seroconversion_rate_matrix[infection_state_to_index({0,1}),infection_state_to_index({1,1})] = baseline_hazards[1] * exp(beta_matrix[2,1]);
        baseline_seroconversion_rate_matrix[infection_state_to_index({0,0}),infection_state_to_index({0,1})] = baseline_hazards[2];
        baseline_seroconversion_rate_matrix[infection_state_to_index({1,0}),infection_state_to_index({1,1})] = baseline_hazards[2] * exp(beta_matrix[1,2]);
        baseline_seroconversion_rate_matrix[infection_state_to_index({0,0}),infection_state_to_index({0,0})] = -baseline_hazards[1] - baseline_hazards[2];
        baseline_seroconversion_rate_matrix[infection_state_to_index({1,0}),infection_state_to_index({1,0})] = -baseline_hazards[2] * exp(beta_matrix[1,2]);
        baseline_seroconversion_rate_matrix[infection_state_to_index({0,1}),infection_state_to_index({0,1})] = -baseline_hazards[1] * exp(beta_matrix[2,1]);
    }

    array[N] real frailties; {
        for (i in 1:N) {
            frailties[i] = exp(log_frailty_deviations[i] * log_frailty_std - 0.5 * log_frailty_std^2); // Individual frailty u_i = exp(σ_u * z_i - σ_u^2/2) where z_i ~ N(0,1)
        }
    }
}

model {
    // Priors
    // target += -log(baseline_hazards); // log(1/λ₀) = -log(λ₀)
    baseline_hazards ~ lognormal(log_baseline_hazard_mean, log_baseline_hazard_scale); // Log-normal prior on baseline hazards
    betas ~ double_exponential(0, beta_scale);
    // target += -log(seroreversion_rates); // TO DO: make this uninformative
    seroreversion_rates ~ normal(0,seroreversion_rate_scale);
    log_frailty_std ~ normal(0,log_frailty_std_scale);

    // Likelihood
    log_frailty_deviations ~ normal(0, 1);
    array[N] real log_lik = rep_array(0.0, N); {
        int obs_idx = 1; // Index for the current observation
        array[K] int prev_serostatus;
        real prev_obs_time;
        int prev_state_index;
        // matrix[num_infection_states,1] prev_infection_state_vector;
        array[K] int next_serostatus;
        real next_obs_time;
        int next_state_index;
        matrix[num_infection_states,1] next_infection_state_vector;
        matrix[num_infection_states,num_infection_states] q_matrix;
        for (i in 1:N) {
            q_matrix = baseline_seroconversion_rate_matrix + seroreversion_rate_matrix * frailties[i];
            prev_serostatus = serostatus[obs_idx,]; // Initial serostatus for the individual
            prev_obs_time = obs_times[obs_idx]; // Initial test time for the individual
            prev_state_index = infection_state_to_index(prev_serostatus);
            next_serostatus = serostatus[obs_idx+1,];
            next_state_index = infection_state_to_index(next_serostatus);
            next_infection_state_vector = rep_matrix(0.0, num_infection_states, 1);
            next_infection_state_vector[next_state_index,1] = 1.0; // Set the initial state vector
            obs_idx += 1;
            for (j in 1:num_obs[i]-1) {
                // Get the current test time and serostatus
                next_serostatus = serostatus[obs_idx];
                next_obs_time = obs_times[obs_idx];
                next_state_index = infection_state_to_index(next_serostatus);
                next_infection_state_vector = rep_matrix(0.0, num_infection_states, 1);
                next_infection_state_vector[next_state_index,1] = 1.0;
                // log_lik[i] += log(transition_matrix[prev_state_index,next_state_index]);
                log_lik[i] += log(
                    matrix_exp_multiply( //this is equivalent to getting the entry [prev_state_index, next_state_index] of the matrix exponential
                        q_matrix * (next_obs_time - prev_obs_time), // q_matrix times time difference between tests
                        next_infection_state_vector
                    )[prev_state_index,1]
                );
                // Update the previous state for the next iteration
                prev_serostatus = next_serostatus;
                prev_obs_time = next_obs_time;
                prev_state_index = next_state_index;
                //prev_infection_state_vector = next_infection_state_vector;
                obs_idx += 1;
            }
        }
    }
    target += log_lik;
}

generated quantities {
    array[N_test] real log_lik_test; {
        real frailty_sample;
        array[N_test, n_frailty_samples] real log_lik_array = rep_array(0.0, N_test, n_frailty_samples); {
            int obs_idx = 1; // Index for the current test
            array[K] int prev_serostatus;
            real prev_obs_time;
            int prev_state_index;
            // matrix[num_infection_states,1] prev_infection_state_vector;
            array[K] int next_serostatus;
            real next_obs_time;
            int next_state_index;
            matrix[num_infection_states,1] next_infection_state_vector;
            matrix[num_infection_states,num_infection_states] q_matrix;

            for (frailty_sample_idx in 1:n_frailty_samples) {
                obs_idx = 1;
                for (i in 1:N_test) {
                    frailty_sample = lognormal_rng(-0.5 * log_frailty_std^2, log_frailty_std); // Sample frailty from the same distribution as in the model
                    q_matrix = baseline_seroconversion_rate_matrix + seroreversion_rate_matrix * frailty_sample;
                    prev_serostatus = serostatus_test[obs_idx,]; // Initial serostatus for the individual
                    prev_obs_time = obs_times_test[obs_idx]; // Initial test time for the individual
                    prev_state_index = infection_state_to_index(prev_serostatus);
                    next_serostatus = serostatus_test[obs_idx+1,];
                    next_state_index = infection_state_to_index(next_serostatus);
                    next_infection_state_vector = rep_matrix(0.0, num_infection_states, 1);
                    next_infection_state_vector[next_state_index,1] = 1.0; // Set the initial state vector
                    obs_idx += 1;
                    for (j in 1:num_obs_test[i]-1) {
                        // Get the current test time and serostatus
                        next_serostatus = serostatus_test[obs_idx];
                        next_obs_time = obs_times_test[obs_idx];
                        next_state_index = infection_state_to_index(next_serostatus);
                        next_infection_state_vector = rep_matrix(0.0, num_infection_states, 1);
                        next_infection_state_vector[next_state_index,1] = 1.0;
                        // log_lik[i] += log(transition_matrix[prev_state_index,next_state_index]);
                        log_lik_array[i, frailty_sample_idx] += log(
                            matrix_exp_multiply( //this is equivalent to getting the entry [prev_state_index, next_state_index] of the matrix exponential
                                q_matrix * (next_obs_time - prev_obs_time), // q_matrix times time difference between tests
                                next_infection_state_vector
                            )[prev_state_index,1]
                        );
                        // Update the previous state for the next iteration
                        prev_serostatus = next_serostatus;
                        prev_obs_time = next_obs_time;
                        prev_state_index = next_state_index;
                        //prev_infection_state_vector = next_infection_state_vector;
                        obs_idx += 1;
                    }
                }
            }
        }
        for (i in 1:N_test) {
            log_lik_test[i] = log_sum_exp(log_lik_array[i,]) - log(n_frailty_samples);
        }
    }
}
