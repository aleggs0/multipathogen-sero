data {
    int<lower=0> N;                         // Number of individuals
    int<lower=0> K;                         // Number of pathogens
    array[N,K] int<lower=0, upper=1> serostatus; // 0 = always negative, 1 = seroconverter
    array[N,K] real time_lower;          // Lower bound of seroconversion time for each pathogen
    array[N,K] real time_upper;          // Upper bound of seroconversion time for each pathogen (will be ignored if serostatus is 0)

    // int<lower=0> n_rows;                    // Number of data rows
    // array[n_rows] int<lower=1, upper=N> person_id; // Individual ID for each test (should be 1-indexed)
    // array[n_rows] real test_time;           // Time of serological test
    // array[n_rows,K] int<lower=0, upper=1> serostatus;    // Serostatus for each pathogen
    // assume no other covariates available
    real <lower=0> time_to_immunity; // timescale for immunity to kick in
                                     // not scientific, just to make the likelihood continuous wrt latent infection times 
    real <lower=0> hazard_ratio_scale; // scale for Laplace prior on hazard ratios
}

transformed data {
    array[K,K] int hazard_ratio_indices; // Array mapping pathogen,pathogen pairs to their indices in hazard_ratios
    {int idx = 0;
    for (i in 1:K) {
        for (j in 1:K) {
            if (i == j) {
                hazard_ratio_indices[i,j] = 0; // Diagonal elements are arbitrary (no interaction)
            } else {
                idx += 1;
                hazard_ratio_indices[i,j] = idx;
            }
        }
    }
    }
    array[N,K] int infection_time_indices; // Index for each individual's infection time for each pathogen
    int len_infection_times = 0;
    for (n in 1:N) {
        for (k in 1:K) {
            if (serostatus[n,k] == 0) {
                infection_time_indices[n,k] = 0;
            } else {
                len_infection_times += 1;
                infection_time_indices[n,k] = len_infection_times;
            }
        }
    }
}

parameters {
    array[K] real<lower=0> baseline_hazards;           // Constant baseline hazard λ₀
    array[K*(K-1)] real<lower=0> hazard_ratios; // Hazard ratios for pathogen, pathogen pair where the pathogens are distinct
    array[len_infection_times] real infection_times; // Infection times for each person, pathogen pair corresponding to a seroconversion
}

// transformed parameters {
//     array[K,K] real hazard_ratio_matrix; // Hazard matrix for pairwise interactions
//     {int idx = 1;
//     for (i in 1:K) {
//         for (j in 1:K) {
//             if (i == j) {
//                 hazard_ratio_matrix[i,j] = 1.0; // Diagonal elements are arbitrary (no interaction)
//             } else {
//             hazard_ratio_matrix[i,j] = hazard_ratio_list[idx];
//             idx += 1;
//             }
//         }
//     }}
// }

model {
    // Priors
    target += -log(baseline_hazards); // log(1/λ₀) = -log(λ₀)
    hazard_ratios ~ double_exponential(0, hazard_ratio_scale);

    // Likelihood
    for (n in 1:N) {
        for (k in 1:K) {
            int serostatus_n_k = serostatus[n,k];
            int infection_time_index_n_k = infection_time_indices[n,k];
            if (serostatus[n,k] == 0) {
                target+= 0; //TODO
            }
            else {
                target+= 0; //TODO
            }
        }
    }
}

generated quantities {
    vector[N] log_lik;
    
    // Initialize log_lik for each person
    for (n in 1:N) {
        log_lik[n] = 0;
    }
    
    // Sum log-likelihood contributions for each person
    for (i in 1:n_rows) {
        real log_surv_prob = - baseline_hazard * (T_R[i] - T_L[i]);
        
        if (event[i] == 1) {
            // Interval: [T_L, T_R], event occurs at T_R
            log_lik[person_id[i]] += log(baseline_hazard) + log_surv_prob;
        } else {
            // Interval: [T_L, T_R), right-censored
            log_lik[person_id[i]] += log_surv_prob;
        }
    }
}