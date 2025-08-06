data {
    int<lower=0> N;                         // Number of individuals
    int<lower=0> P;                         // Number of covariates
    int<lower=0> n_rows;                    // Number of data rows
    vector[n_rows] T_L;                     // Left truncation times
    vector[n_rows] T_R;                     // Right endpoint times (event or censoring)
    array[n_rows] int<lower=0, upper=1> event;    // Event indicator: 1 if event, 0 if right-censored
    array[n_rows] int<lower=1, upper=N> person_id; // Individual ID per row (should be 1-indexed)
    matrix[n_rows, P] X;                    // Covariate matrix
    int<lower=0, upper=2> beta_prior_setting; // 0 for no prior, 1 for Laplace prior, 2 for spike and slab
    real<lower=0> laplace_scale; // scale for Laplace prior when beta_prior_setting == 1
    real<lower=0> slab_scale;    // scale for slab component when beta_prior_setting == 2
    real<lower=0> spike_scale;   // scale for spike component (small) when beta_prior_setting == 2
    real<lower=0, upper=1> pi;   // prior probability of inclusion when beta_prior_setting == 2
}

parameters {
    real<lower=0> baseline_hazard;         // Constant baseline hazard λ₀
    vector[P] beta_raw;                        // Regression coefficients (only for Laplace prior)
}

transformed parameters {
   vector[P] beta;
    if (beta_prior_setting == 0) { // No prior
        beta = rep_vector(0, P);
    } else  {
        beta = beta_raw;
    }
}

model {
    // Priors
    target += -log(baseline_hazard); // log(1/λ₀) = -log(λ₀)
    if (beta_prior_setting == 2) {
        for (p in 1:P) {
            target += log_mix(pi, normal_lpdf(beta_raw[p] | 0, slab_scale), normal_lpdf(beta_raw[p] | 0, spike_scale));
        }
    } else {
        beta_raw ~ double_exponential(0, laplace_scale); // Laplace prior (shrinkage) on β, vectorized
    }
    // Likelihood
    for (i in 1:n_rows) {
        real linpred = dot_product(X[i], beta);
        real individual_hazard = baseline_hazard * exp(linpred);
        real log_surv_prob = - individual_hazard * (T_R[i] - T_L[i]);

        if (event[i] == 1) {
            // Interval: [T_L, T_R], event occurs at T_R
            target += log(individual_hazard) + log_surv_prob;
        } else {
            // Interval: [T_L, T_R), right-censored
            target += log_surv_prob;
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
        real linpred = dot_product(X[i], beta);
        real individual_hazard = baseline_hazard * exp(linpred);
        real log_surv_prob = - individual_hazard * (T_R[i] - T_L[i]);
        
        if (event[i] == 1) {
            // Interval: [T_L, T_R], event occurs at T_R
            log_lik[person_id[i]] += log(individual_hazard) + log_surv_prob;
        } else {
            // Interval: [T_L, T_R), right-censored
            log_lik[person_id[i]] += log_surv_prob;
        }
    }
}