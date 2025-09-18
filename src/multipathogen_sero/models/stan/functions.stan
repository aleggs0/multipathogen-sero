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
}