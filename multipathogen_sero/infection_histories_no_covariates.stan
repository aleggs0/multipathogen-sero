data {
    int<lower=0> N;                         // Number of individuals
    int<lower=0> P;                         // Number of covariates
    int<lower=0> n_rows;                    // Number of data rows
    vector[n_rows] T_L;                     // Left truncation times
    vector[n_rows] T_R;                     // Right endpoint times (event or censoring)
    array[n_rows] int<lower=0, upper=1> event;    // Event indicator: 1 if event, 0 if right-censored
    array[n_rows] int<lower=1, upper=N> person_id // Individual ID per row (should be 1-indexed)
}

parameters {
    real<lower=0> baseline_hazard;         // Constant baseline hazard λ₀
}

model {
    // Priors
    
    target += -log(baseline_hazard); // log(1/λ₀) = -log(λ₀)

    // Likelihood
    for (i in 1:n_rows) {
        real log_surv_prob = - baseline_hazard * (T_R[i] - T_L[i]);

        if (event[i] == 1) {
            // Interval: [T_L, T_R], event occurs at T_R
            target += log(baseline_hazard) + log_surv_prob;
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