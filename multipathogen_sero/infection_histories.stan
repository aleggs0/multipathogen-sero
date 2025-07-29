data {
    int<lower=0> N;                         // Number of individuals
    int<lower=0> P;                         // Number of covariates
    int<lower=0> n_rows;                    // Number of data rows
    vector[n_rows] T_L;                     // Left truncation times
    vector[n_rows] T_R;                     // Right endpoint times (event or censoring)
    array[n_rows] int<lower=0, upper=1> event;    // Event indicator: 1 if event, 0 if right-censored
    vector<lower=0, upper=N>[n_rows] person_id; // Individual ID per row (agnostic to whether zero-indexed)
    matrix[n_rows, P] X;                    // Covariate matrix
}

parameters {
    real<lower=0> baseline_hazard;         // Constant baseline hazard λ₀
    vector[P] beta;                        // Regression coefficients
}

model {
    // Priors
    
    target += -log(baseline_hazard); // log(1/λ₀) = -log(λ₀)
    beta ~ double_exponential(0, 0.5);      // Laplace prior (shrinkage) on β

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