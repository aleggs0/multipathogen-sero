data { // TODO: code the censoring times in a smarter way
    int<lower=1> N;                         // Number of individuals
    array[N] real birth_times;               // Birth time for each individual
    int<lower=2,upper=2> K;                         // Number of pathogens
    array[N,K] int<lower=0, upper=1> gets_infected; // 0 = always negative, 1 = seroconverter
    int<lower=0> num_infections; // Number of infection times (seroconversions)
    int<lower=0> num_noninfections; // Number of non-infection individuals-pathogen pairs (seronegative)
    array[N,K] int<lower=1, upper=N*K> lookup_indices; // Index to look up individual, pathogen infection times in their respective (1-indexed) arrays
    array[num_infections] real lower_bound_times; // Lower bound of seroconversion time
    array[num_infections] real upper_bound_times; // Upper bound of seroconversion time
    array[num_infections] int<lower=1, upper=N> infection_person_ids; // Individual ID for each seroconversion (should be 1-indexed)
    array[num_infections] int<lower=1, upper=K> infection_pathogen_ids; // Pathogen ID for each seroconversion (should be 1-indexed)
    array[num_noninfections] real censoring_times; // Lower bound of seroconversion time
    array[num_noninfections] int<lower=1, upper=N> noninfection_person_ids; // Individual ID for each seronegative individual (should be 1-indexed)
    array[num_noninfections] int<lower=1, upper=K> noninfection_pathogen_ids; // Pathogen ID for each seronegative individual (should be 1-indexed)
    real <lower=0> time_to_immunity; // timescale for immunity to kick in
                                     // not scientific, just to make the likelihood continuous wrt latent infection times 
    real <lower=0> beta_scale; // scale for Laplace prior on log hazard ratios
    int<lower=1> n_frailty_samples; // Number of frailty samples to draw in generated quantities
    real <lower=0> log_frailty_std_scale; // scale for half-normal prior on scale of individual frailties, sigma_u
}

transformed data {
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
    // array[N,K] int infection_time_indices; // Index for each individual's infection time for each pathogen
    // int len_infection_times = 0;
    // for (n in 1:N) {
    //     for (k in 1:K) {
    //         if (serostatus[n,k] == 0) {
    //             infection_time_indices[n,k] = 0;
    //         } else {
    //             len_infection_times += 1;
    //             infection_time_indices[n,k] = len_infection_times;
    //         }
    //     }
    // }
/*
    array[num_infections] real infection_times; // Latent infection times for seroconverters
    for (i in 1:num_infections) {
        infection_times[i] = (upper_bound_times[i] + lower_bound_times[i]) / 2;
    }*/
}

parameters {
    array[K] real<lower=0> baseline_hazards;    // Constant baseline hazard λ₀
    array[K*(K-1)] real betas;                  // Log hazard ratios for pathogen, pathogen pair where the pathogens are distinct
    array[num_infections] real<lower=lower_bound_times, upper=upper_bound_times> infection_times; // Latent infection times for seroconverters
    array[N] vector[K] log_frailty_direction; // Log frailty terms for each pathogen and individual
    vector<lower=0>[K] log_frailty_scale;      // log frailty scale
    cholesky_factor_corr[K] L;         // cholesky representation of log frailty correlation
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
}

model {
    // Priors
    target += -log(baseline_hazards); // log(1/λ₀) = -log(λ₀)
    betas ~ double_exponential(0, beta_scale);
    log_frailty_scale ~ normal(0, log_frailty_std_scale);
    L ~ lkj_corr_cholesky(log_frailty_corr_shape); // want 1 but had divergences
    // Latent
    for (i in 1:N) {
        log_frailty_direction[i] ~ multi_normal_cholesky(rep_vector(0, K), L); // Log frailty terms
    }
    // log_frailty ~ multi_normal_cholesky(rep_vector(0, K), diag_pre_multiply(log_frailty_scale, L)); // Log frailty terms
    // Likelihood
    array[N] real log_lik = rep_array(0.0, N);
    for (n in 1:N) {
        real indiv_birth_time = birth_times[n];
        array[K] int indiv_gets_infected = gets_infected[n];
        int indiv_num_infections = sum(indiv_gets_infected);
        array[K] int indiv_lookup_indices = lookup_indices[n];
        array[indiv_num_infections] int indiv_infection_indices = boolean_index(indiv_lookup_indices, indiv_gets_infected);
        array[K-indiv_num_infections] int indiv_noninfection_indices; {
            int counter = 0;
            for (k in 1:K) {
                if (indiv_gets_infected[k] == 0) {
                    counter += 1;
                    indiv_noninfection_indices[counter] = indiv_lookup_indices[k];
                }
            }
        }
        array[K-indiv_num_infections] int indiv_noninfection_pathogen_ids = noninfection_pathogen_ids[indiv_noninfection_indices];
        array[indiv_num_infections] int indiv_infection_pathogen_ids = infection_pathogen_ids[indiv_infection_indices];
        array[indiv_num_infections] real indiv_infection_times = infection_times[indiv_infection_indices];
        vector[indiv_num_infections] indiv_infection_time_vector = to_vector(indiv_infection_times);
        vector[indiv_num_infections] indiv_infection_time_vector_sorted = sort_asc(indiv_infection_time_vector);
        array[indiv_num_infections] int indiv_infection_sorted_indices = sort_indices_asc(indiv_infection_time_vector);
        array[indiv_num_infections] int indiv_infection_pathogen_ids_sorted = indiv_infection_pathogen_ids[indiv_infection_sorted_indices];
        vector[K] indiv_hazards = to_vector(baseline_hazards) .* exp(log_frailty_direction[n] .* log_frailty_scale);
        for (k in 1:indiv_num_infections) {
            if (k==1) {
                log_lik[n] += -sum(indiv_hazards)*(indiv_infection_time_vector_sorted[1] - indiv_birth_time); // Survival probability to first infection time
            } else {
                log_lik[n] += -sum(indiv_hazards) * (indiv_infection_time_vector_sorted[k] - indiv_infection_time_vector_sorted[k-1]); // Survival probability to next infection time
            }
            int pathogen_id = indiv_infection_pathogen_ids_sorted[k];
            log_lik[n] += log(smoothed_individual_hazard(
                baseline_hazards[pathogen_id],
                beta_matrix[indiv_infection_pathogen_ids, pathogen_id],
                indiv_infection_times,
                time_to_immunity,
                indiv_infection_time_vector_sorted[k]
            ));
            indiv_hazards[pathogen_id] = 0;
            for (l in 1:K) {
                indiv_hazards[l] *= exp(beta_matrix[pathogen_id, l]);
            }
            
        }
        if (num_elements(indiv_noninfection_indices) > 0) {
            real interval_start = indiv_birth_time;
            if (indiv_num_infections > 0) {
                interval_start = indiv_infection_time_vector_sorted[indiv_num_infections];
            }
            log_lik[n] += -sum(indiv_hazards) * (censoring_times[indiv_noninfection_indices[1]] - interval_start); // Survival probability to the (common) censoring time
        }
    }
    target += log_lik; // Sum log-likelihood contributions for all individuals
}

generated quantities {
    // Likelihood
    array[N] real log_lik = rep_array(0.0, N);
    {
        for (n in 1:N) {
            real indiv_birth_time = birth_times[n];
            array[K] int indiv_gets_infected = gets_infected[n];
            int indiv_num_infections = sum(indiv_gets_infected);
            array[K] int indiv_lookup_indices = lookup_indices[n];
            array[indiv_num_infections] int indiv_infection_indices = boolean_index(indiv_lookup_indices, indiv_gets_infected);
            array[K-indiv_num_infections] int indiv_noninfection_indices; {
                int counter = 0;
                for (k in 1:K) {
                    if (indiv_gets_infected[k] == 0) {
                        counter += 1;
                        indiv_noninfection_indices[counter] = indiv_lookup_indices[k];
                    }
                }
            }
            array[K-indiv_num_infections] int indiv_noninfection_pathogen_ids = noninfection_pathogen_ids[indiv_noninfection_indices];
            array[indiv_num_infections] int indiv_infection_pathogen_ids = infection_pathogen_ids[indiv_infection_indices];
            array[indiv_num_infections] real indiv_infection_times = infection_times[indiv_infection_indices];
            vector[indiv_num_infections] indiv_infection_time_vector = to_vector(indiv_infection_times);
            vector[indiv_num_infections] indiv_infection_time_vector_sorted = sort_asc(indiv_infection_time_vector);
            array[indiv_num_infections] int indiv_infection_sorted_indices = sort_indices_asc(indiv_infection_time_vector);
            array[indiv_num_infections] int indiv_infection_pathogen_ids_sorted = indiv_infection_pathogen_ids[indiv_infection_sorted_indices];
            vector[K] indiv_hazards = to_vector(baseline_hazards) .* exp(log_frailty_direction[n] .* log_frailty_scale);
            for (k in 1:indiv_num_infections) {
                if (k==1) {
                    log_lik[n] += -sum(indiv_hazards)*(indiv_infection_time_vector_sorted[1] - indiv_birth_time); // Survival probability to first infection time
                } else {
                    log_lik[n] += -sum(indiv_hazards) * (indiv_infection_time_vector_sorted[k] - indiv_infection_time_vector_sorted[k-1]); // Survival probability to next infection time
                }
                int pathogen_id = indiv_infection_pathogen_ids_sorted[k];
                log_lik[n] += log(smoothed_individual_hazard(
                    baseline_hazards[pathogen_id],
                    beta_matrix[indiv_infection_pathogen_ids, pathogen_id],
                    indiv_infection_times,
                    time_to_immunity,
                    indiv_infection_time_vector_sorted[k]
                ));
                indiv_hazards[pathogen_id] = 0;
                for (l in 1:K) {
                    indiv_hazards[l] *= exp(beta_matrix[pathogen_id, l]);
                }
                
            }
            if (num_elements(indiv_noninfection_indices) > 0) {
                real interval_start = indiv_birth_time;
                if (indiv_num_infections > 0) {
                    interval_start = indiv_infection_time_vector_sorted[indiv_num_infections];
                }
                log_lik[n] += -sum(indiv_hazards) * (censoring_times[indiv_noninfection_indices[1]] - interval_start); // Survival probability to the (common) censoring time
            }
        }
    }
    corr_matrix[K] log_frailty_corr = multiply_lower_tri_self_transpose(L); // Convert Cholesky factor to correlation matrix
}