#include functions.stan


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

    real <lower=0> baseline_hazard_scale; // Scale for half-normal prior on baseline hazards
    real <lower=0> beta_scale; // scale for Laplace prior on log hazard ratios
    real <lower=0> seroreversion_rate_scale; // scale for half-normal prior on seroreversion rates
}

transformed data {
    int num_lags = num_obs_total - N; // Total number of lags (time between tests) across all individuals
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
}
parameters {
    array[K] real<lower=0> baseline_hazards;     // Constant baseline hazard λ₀
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
}

model {
    // Priors
    // target += -log(baseline_hazards); // log(1/λ₀) = -log(λ₀)
    baseline_hazards ~ normal(0, baseline_hazard_scale); // half-normal prior on baseline hazards
    betas ~ double_exponential(0, beta_scale);
    // target += -log(seroreversion_rates); // TO DO: make this uninformative
    seroreversion_rates ~ normal(0,seroreversion_rate_scale);

    // Likelihood
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
        q_matrix = baseline_seroconversion_rate_matrix + seroreversion_rate_matrix;
        for (i in 1:N) {
            prev_serostatus = serostatus[obs_idx,]; // Initial serostatus for the individual
            prev_obs_time = obs_times[obs_idx]; // Initial test time for the individual
            prev_state_index = infection_state_to_index(prev_serostatus);
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
    array[num_lags] real log_lik; {
        int obs_idx = 1; // Index for the current observation
        int lag_idx = 1; // Index for the current lag (offsets from obs_idx by 1 after each indiv)
        array[K] int prev_serostatus;
        real prev_obs_time;
        int prev_state_index;
        // matrix[num_infection_states,1] prev_infection_state_vector;
        array[K] int next_serostatus;
        real next_obs_time;
        int next_state_index;
        matrix[num_infection_states,1] next_infection_state_vector;
        matrix[num_infection_states,num_infection_states] q_matrix;
        q_matrix = baseline_seroconversion_rate_matrix + seroreversion_rate_matrix;
        for (i in 1:N) {
            prev_serostatus = serostatus[obs_idx,]; // Initial serostatus for the individual
            prev_obs_time = obs_times[obs_idx]; // Initial test time for the individual
            prev_state_index = infection_state_to_index(prev_serostatus);
            obs_idx += 1;
            for (j in 1:num_obs[i]-1) {
                // Get the current test time and serostatus
                next_serostatus = serostatus[obs_idx];
                next_obs_time = obs_times[obs_idx];
                next_state_index = infection_state_to_index(next_serostatus);
                next_infection_state_vector = rep_matrix(0.0, num_infection_states, 1);
                next_infection_state_vector[next_state_index,1] = 1.0;
                // log_lik[i] += log(transition_matrix[prev_state_index,next_state_index]);
                log_lik[lag_idx] = log(
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
                lag_idx += 1;
            }
        }
    }
    array[N_test] real log_lik_test = rep_array(0.0, N_test);
    {
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
        q_matrix = baseline_seroconversion_rate_matrix + seroreversion_rate_matrix;
        for (i in 1:N_test) {
            prev_serostatus = serostatus_test[obs_idx,]; // Initial serostatus for the individual
            prev_obs_time = obs_times_test[obs_idx]; // Initial test time for the individual
            prev_state_index = infection_state_to_index(prev_serostatus);
            obs_idx += 1;
            for (j in 1:num_obs_test[i]-1) {
                // Get the current test time and serostatus
                next_serostatus = serostatus_test[obs_idx];
                next_obs_time = obs_times_test[obs_idx];
                next_state_index = infection_state_to_index(next_serostatus);
                next_infection_state_vector = rep_matrix(0.0, num_infection_states, 1);
                next_infection_state_vector[next_state_index,1] = 1.0;
                // log_lik[i] += log(transition_matrix[prev_state_index,next_state_index]);
                log_lik_test[i] += log(
                    matrix_exp_multiply( //this is equivalent to getting the entry [prev_state_index, next_state_index] of the matrix exponential
                        q_matrix * (next_obs_time - prev_obs_time), // q_matrix times time difference between tests
                        next_infection_state_vector
                    )[prev_state_index,1]
                );
                prev_serostatus = next_serostatus;
                prev_obs_time = next_obs_time;
                prev_state_index = next_state_index;
                obs_idx += 1;
            }
        }
    }
}