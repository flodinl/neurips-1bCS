import numpy as np
import math
import gurobipy


def sign_measure(A, v):
    ''' Returns the product Av thresholded into a -1/+1 vector (ndarray).
    '''
    return np.sign(A.dot(v))

def gen_rand_bern_matrix(n, m, p):
    ''' Returns an m x n matrix where each entry is 1 with probability p and 0 otherwise.
    '''
    matrix = np.random.rand(m, n)
    small_indices = (matrix <= p)
    large_indices = (matrix > p)
    matrix[small_indices] = 1
    matrix[large_indices] = 0
    return matrix

def gen_rand_real_vector(n, k):
    ''' Returns a randomly generated k-sparse vector on the unit sphere, by first
    picking the k coordinates uniformly at random, then drawing a Gaussian for each
    coordinate and rescaling so that the norm is 1.
    '''
    support = np.random.choice(n, size=k, replace=False)
    x = np.zeros(n)
    x[support] = np.random.normal(size=k, scale=1)
    rescaled_x = x / np.linalg.norm(x)
    return rescaled_x

def dropout_known_sparsity(x, k):
    ''' Given a vector x, return a vector with all but the k largest entries of x set to 0.
    '''
    dropout_x = np.copy(x)
    sorted_x = np.sort(np.absolute(dropout_x))
    kplus1_largest = sorted_x[-(k + 1)]
    small_indices = np.absolute(dropout_x) <= kplus1_largest
    dropout_x[small_indices] = 0
    return dropout_x

def group_testing_decode(A, y):
    ''' Returns a set of indices corresponding to the coordinates of the signal that cannot be ruled
    out from performing a group testing decoding of A and y (i.e. all coordinates that do not
    appear in any test with a 0 result). If x is orthogonal to a row of A, there can be false negatives, otherwise
    there will be only false positives.
    '''
    n = A.shape[1]
    superset = np.arange(n)
    for index, entry in enumerate(y):
        if entry == 0:
            neg_indices = (A[index] != 0)
            superset[neg_indices] = -1
    final_superset = np.delete(superset, np.where(superset == -1))
    return final_superset

def biht_reals(A, y, k, step_size=0.001, max_iterations=1000):
    ''' Binary Iterative Hard Thresholding algorithm for reconstruction of real unit vector from sign measurements,
    implemented based on description in JLBB '13. Interleaves steps of standard gradient descent for l_2 loss
    with steps of thresholding all but largest k entries down to 0. A is the measurement matrix, y is the 
    vector of sign measurements, k is the sparsity, and step_size parameter determines the step size of the
    gradient descent steps. Solution is normalized to have norm 1 at the end of all iterations 
    (but not between iterations).
    '''

    m = A.shape[0]
    n = A.shape[1]
    tau = step_size

    a, x = np.zeros(n), np.zeros(n)
    converged = False
    iteration_num = 1
    while (not converged and iteration_num <= max_iterations):
        prev_a = a
        prev_x = x
        a = x + (tau / 2) * np.matmul(A.T, (y - sign_measure(A, x)))
        if x.size > k:
            x = dropout_known_sparsity(a, k)
        else:
            x = a

        if iteration_num != 1 and np.linalg.norm(x / np.linalg.norm(x) - prev_x / np.linalg.norm(prev_x)) < 1e-5:
            converged = True

        iteration_num += 1

    if not converged:
        # print("terminated without convergence after iteration #", str(max_iterations), "\n")
        pass

    return x / np.linalg.norm(x)

def run_biht_experiment(n, k, m_values, num_trials):
    ''' For each value of m in m_values, perform an experiment where for num_trials many trials, 
    a signal of sparsity k
    and length n is generated uniformly at random to be the true signal, then BIHT is run to
    try and construct a sparse solution. Results of the experiments are written to a text file in the same
    directory as this code
    "results_biht_n100_k5.txt" where the numbers after n and k are the parameters passed in.
    '''

    total_error_array = np.zeros(len(m_values))
    for m_index, m in enumerate(m_values):
        for trial in range(num_trials):
            A = np.random.normal(size=(m, n), scale=1)
            x = gen_rand_real_vector(n, k)
            y = sign_measure(A, x)

            reconstructed_x = biht_reals(A, y, k)

            error = np.linalg.norm(x - reconstructed_x)
            total_error_array[m_index] += error
        print("for m=", str(m), "average error of all gaussian exp. was", total_error_array[m_index] / num_trials)

    avg_error_array = total_error_array / num_trials
    f = open("results_biht_n" + str(n) + "_k" + str(k) + ".txt", mode='w')
    f.write("m_values were " + str(list(m_values)) + "\n")
    f.write("the following is average error over " + str(num_trials) + " trials with no rows used for recovering superset \n")
    f.write(str(list(avg_error_array)))
    f.close()

def run_superset_experiment(n, k, m_values, num_trials):
    ''' For each value of m in m_values, perform an experiment of num_trials many trials, where in each trial
    a uniformly random k-sparse vector of length n is generated on the sphere. In each trial, a superset of the signal
    support is recovered using a Bern(1/(k+1)) 0/1 matrix and group testing decoding, then the remainder of the m
    measurements are used for a Gaussian matrix with BIHT to recover the signal within the superset.
    signal within the support superset. This process is repeated using k log_10(n), 2k log_10(n), 3k log_10(n), and 
    4k log_10(n) measurements in the Bernoulli matrix, with the remainder of the m measurements used for the Gaussian matrix.
    Results of the experiments are written to a text file in the same directory as this code
    "results_superset_n100_k5.txt" where the numbers after n and k are the parameters passed in.
    '''

    m1_values_length = 4
    total_error_array = np.zeros((len(m_values), m1_values_length))
    superset_sizes_array = np.zeros((len(m_values), m1_values_length))
    for m_index, m in enumerate(m_values):
        
        # for each entry of m1_values, a sub-experiment is performed using that many measurements
        # in the superset matrix, and m - m1 measurements in the Gaussian matrix
        # these values perform well in practice for a fairly wide range of inputs, but may need
        # more sophisticated tuning in some cases 
        m1_values = [int(k * math.log(n, 10)), int(2 * k * math.log(n, 10)),
        	int(3 * k * math.log(n, 10)), int(4 * k * math.log(n, 10))]
        
        for m1_index, m1 in enumerate(m1_values):
            for trial in range(num_trials):
                x = gen_rand_real_vector(n, k)
                m2 = m - m1
                
                # experimentally this value performs best regardless of how other parameters are set
                bern_prob = 1 / (k + 1)

                A1 = gen_rand_bern_matrix(n, m1, bern_prob)
                y1 = sign_measure(A1, x)
                support_superset = group_testing_decode(A1, y1)
                
                restricted_x = x[support_superset]
                A2 = np.random.normal(size=(m2, restricted_x.size), scale=1)
                y2 = sign_measure(A2, restricted_x)
                reconstructed_restricted_x = biht_reals(A2, y2, k)
                reconstructed_x = np.zeros(n)
                reconstructed_x[support_superset] = reconstructed_restricted_x
                error = np.linalg.norm(x - reconstructed_x)
                total_error_array[m_index][m1_index] += error
                superset_sizes_array[m_index][m1_index] += support_superset.size
            print("for m=", str(m), "m1=", str(m1), "average error of superset exp. was", total_error_array[m_index][m1_index] / num_trials)

    avg_error_array = total_error_array / num_trials
    avg_superset_size_array = superset_sizes_array / num_trials

    f = open("results_superset_n" + str(n) + "_k" + str(k) + ".txt", mode='w')
    f.write("m_values were " + str(list(m_values)) + "\n")
    f.write("the following is average error over " + str(num_trials) + " trials with respectively " + str(m1_values) + " rows used for recovering superset, followed by avg. superset sizes \n")
    for index, row in enumerate(avg_error_array.T):
        f.write(str(list(row)) + "\n")
        f.write(str(list(avg_superset_size_array.T[index])) + "\n")
    f.close()

    return avg_superset_size_array


def main():
    np.random.seed(0)

    n = 20
    k_values = [5, 10]
    m_values = [100, 200]
    num_trials = 100
    for k in k_values:
        run_biht_experiment(n, k, m_values, num_trials)
        run_superset_experiment(n, k, m_values, num_trials)
        

if __name__ == '__main__':
    main()