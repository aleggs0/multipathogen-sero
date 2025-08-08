functions{
    real individual_hazard(
        real baseline_hazard,
        array[] real relevant_betas,
        array[] real relevant_infection_times,
        real time_to_immunity,
        real time
    ) {
        real hazard = baseline_hazard;
        // Assert that relevant_betas and relevant_infection_times are the same length
        if (num_elements(relevant_betas) != num_elements(relevant_infection_times)) {
            reject("relevant_betas and relevant_infection_times must have the same length.");
        }
        for (k in 1:num_elements(relevant_betas)) {
            if (time >= relevant_infection_times[k]) {
                // Want to do the following but it's not differentiable:
                // hazard *= exp(relevant_betas[k]);
                // so instead:
                // hazard *= exp(relevant_betas[k] * min([1.0, (time - relevant_infection_times[k])/ time_to_immunity]));
                hazard *= 1; //TODO: switch back
            }
        }
        print(hazard);
        return 0.1;
    }

    real individual_hazard_integrand(
        real x, // time
        real xc,
        array[] real params, // baseline_hazard and relevant_betas and relevant_infection_times
        array[] real x_r, // time_to_immunity
        array[] int x_i
    ) {
        real baseline_hazard = params[1];
        int num_relevant_pathogens = (num_elements(params) - 1) %/% 2;
        array[num_relevant_pathogens] real relevant_betas = params[2:num_relevant_pathogens+1];
        array[num_relevant_pathogens] real relevant_infection_times = params[num_relevant_pathogens+2:num_relevant_pathogens*2+1];
        return individual_hazard(
            baseline_hazard,
            relevant_betas,
            relevant_infection_times,
            x_r[1], // time_to_immunity
            x // time
        );
    }

    array[] real concat_params(real baseline_hazard, array[] real relevant_betas, array[] real relevant_infection_times) {
        array [1 + num_elements(relevant_betas) + num_elements(relevant_infection_times)] real params;
        params[1] = baseline_hazard;
        for (i in 1:num_elements(relevant_betas)) {
            params[i + 1] = relevant_betas[i];
        }
        for (i in 1:num_elements(relevant_infection_times)) {
            params[i + num_elements(relevant_betas) + 1] = relevant_infection_times[i];
        }
        return params;
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
}

data {
    int<lower=1> N;                         // Number of individuals
    array[N] real birth_times;               // Birth time for each individual
    int<lower=1> K;                         // Number of pathogens
    array[N,K] int<lower=0, upper=1> gets_infected; // 0 = always negative, 1 = seroconverter
    int<lower=0> num_infections; // Number of infection times (seroconversions)
    int<lower=0> num_noninfections; // Number of uninfected individuals (seronegative)
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
    real <lower=0> relative_tolerance; // Relative tolerance for numerical integration (the default caused exceptions)
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
}

transformed parameters {
    array[K,K] real beta_matrix; // Hazard matrix for pairwise interactions
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
    
    // Likelihood
    array[N] real log_lik = rep_array(0.0, N);
    for (n in 1:N) {
        real indiv_birth_time = birth_times[n];
        array[K] int indiv_gets_infected = gets_infected[n];
        int indiv_num_infections = sum(indiv_gets_infected);
        array[K] int indiv_lookup_indices = lookup_indices[n];
        array[indiv_num_infections] int indiv_infection_indices = boolean_index(indiv_lookup_indices, indiv_gets_infected);
        array[indiv_num_infections] int indiv_infection_pathogen_ids = infection_pathogen_ids[indiv_infection_indices];
        array[indiv_num_infections] real indiv_infection_times = infection_times[indiv_infection_indices];
        
        for (k in 1:K) {
            if (indiv_gets_infected[k] == 0) { // log probability of surviving to time_lower
                log_lik[n] += -integrate_1d(
                    individual_hazard_integrand, 
                    indiv_birth_time,
                    censoring_times[indiv_lookup_indices[k]],
                    concat_params(
                        baseline_hazards[k],
                        beta_matrix[k, indiv_infection_pathogen_ids],
                        indiv_infection_times
                    ),
                    {time_to_immunity}, 
                    {-1}, // unused input
                    relative_tolerance
                );
            } else { // log probability density of seroconversion and infection_time
                log_lik[n] += -integrate_1d(
                    individual_hazard_integrand, 
                    indiv_birth_time,
                    infection_times[indiv_lookup_indices[k]],
                    concat_params(
                        baseline_hazards[k],
                        beta_matrix[k, indiv_infection_pathogen_ids],
                        indiv_infection_times
                    ),
                    {time_to_immunity}, 
                    {-1}, // unused input
                    relative_tolerance
                ) + log(individual_hazard(
                    baseline_hazards[k],
                    beta_matrix[k, indiv_infection_pathogen_ids],
                    indiv_infection_times,
                    time_to_immunity,
                    infection_times[indiv_lookup_indices[k]]
                ));
            }
        }
    }
    target += log_lik; // Sum log-likelihood contributions for all individuals
}

generated quantities {
    // Likelihood
    array[N] real log_lik = rep_array(0.0, N); {
        for (n in 1:N) {
            real indiv_birth_time = birth_times[n];
            array[K] int indiv_gets_infected = gets_infected[n];
            int indiv_num_infections = sum(indiv_gets_infected);
            array[K] int indiv_lookup_indices = lookup_indices[n];
            array[indiv_num_infections] int indiv_infection_indices = boolean_index(indiv_lookup_indices, indiv_gets_infected);
            array[indiv_num_infections] int indiv_infection_pathogen_ids = infection_pathogen_ids[indiv_infection_indices];
            array[indiv_num_infections] real indiv_infection_times = infection_times[indiv_infection_indices];
            
            for (k in 1:K) {
                if (indiv_gets_infected[k] == 0) { // log probability of surviving to time_lower
                    log_lik[n] += -integrate_1d(
                        individual_hazard_integrand, 
                        indiv_birth_time,
                        censoring_times[indiv_lookup_indices[k]],
                        concat_params(
                            baseline_hazards[k],
                            beta_matrix[k, indiv_infection_pathogen_ids],
                            indiv_infection_times
                        ),
                        {time_to_immunity}, 
                        {-1}, // unused input
                        relative_tolerance
                    );
                }
                else { // log probability density of seroconversion and infection_time
                    log_lik[n] += -integrate_1d(
                        individual_hazard_integrand, 
                        indiv_birth_time,
                        infection_times[indiv_lookup_indices[k]],
                        concat_params(
                            baseline_hazards[k],
                            beta_matrix[k, indiv_infection_pathogen_ids],
                            indiv_infection_times
                        ),
                        {time_to_immunity}, 
                        {-1}, // unused input
                        relative_tolerance
                    ) + log(individual_hazard(
                        baseline_hazards[k],
                        beta_matrix[k, indiv_infection_pathogen_ids],
                        indiv_infection_times,
                        time_to_immunity,
                        infection_times[indiv_lookup_indices[k]]
                    ));
                }
            }
        }
    }
}